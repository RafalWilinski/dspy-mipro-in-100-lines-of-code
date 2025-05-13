import { generateObject, generateText, type LanguageModelV1 } from "ai";
import { z } from "zod";
import { extractMeetingsFromEmail } from "./extractMeetingsFromEmail";
import type { DatasetRow, OptimizableFunction } from "./types";
import { arrayOfObjectsScorer } from "./scorers";

async function generateDataSummary<Input, Output>(
  dataset: DatasetRow<Input, Output>[],
  sampleSize: number,
  model: LanguageModelV1
) {
  console.log(`Generating data summary for ${sampleSize} samples`);
  const sample = selectRandomRowsFromDataset(dataset, sampleSize);
  const system = `Analyze the following dataset examples and provide a summary of key patterns, input-output relationships, and any specific challenges the data presents. Focus on what makes a good answer and what patterns should be followed.`;
  const prompt = `${sample.map((row) => JSON.stringify(row)).join("\n")}`;

  return (
    await generateText({
      system,
      model,
      prompt,
    })
  ).text;
}

export async function proposeInstructionCandidates<Input, Output>(
  prompt: string,
  dataset: DatasetRow<Input, Output>[],
  numCandidates: number,
  numSamples: number,
  model: LanguageModelV1
): Promise<string[]> {
  console.log(`Generating ${numCandidates} instruction candidates`);
  const dataContext = await generateDataSummary(dataset, numSamples, model);
  const system = `Create ${numCandidates} high-quality instruction for an AI model performing the task described below.
    
  ${dataContext ? `<data-context>\n${dataContext}\n</data-context>\n\n` : ""}
  
Your task is to craft a clear, effective variation of the following instruction that will help the AI model generate accurate outputs for this task.
  
The instruction should be detailed enough to guide the model but not overly prescriptive or restrictive. Focus on what makes a good response rather than listing exact steps.`;

  const result = await generateObject({
    system,
    model,
    prompt,
    output: "array",
    schema: z.object({ newPrompt: z.string() }),
  });
  return result.object.map((o) => o.newPrompt);
}

export async function evaluate<Input, Output>(
  instruction: string,
  fewShotExamples: DatasetRow<Input, Output>[],
  evalDataset: DatasetRow<Input, Output>[],
  fn: OptimizableFunction<Input, Output>,
  scorer: (output: Output, expected: Output) => number
) {
  let score = 0;
  console.log(`Evaluating: ${instruction}`);
  for (const row of evalDataset) {
    const output = await fn(row.input, fewShotExamples, instruction);
    const expected = row.output;
    score += scorer(output, expected);
  }

  return {
    score: evalDataset.length === 0 ? 0 : score / evalDataset.length,
    instruction,
    fewShotExamples,
  };
}

/**
 * Try to find {numExamples} examples that give us correct results already
 */
export async function bootstrapFewShotExamples<Input, Output>(
  prompt: string,
  dataset: DatasetRow<Input, Output>[],
  numExamples: number,
  fn: OptimizableFunction<string, Output>,
  scorer: (output: Output, expected: Output) => number
) {
  const examples: DatasetRow<Input, Output>[] = [];
  for (const row of dataset) {
    const output = await fn(row.input as string, [], prompt);
    const score = scorer(output, row.output);
    if (score === 1) {
      examples.push(row);
      if (examples.length >= numExamples) {
        break;
      }
    }
  }
  return examples;
}

export function selectRandomRowsFromDataset<Input, Output>(
  dataset: DatasetRow<Input, Output>[],
  maxExamples: number
): DatasetRow<Input, Output>[] {
  const shuffled = [...dataset].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(maxExamples, shuffled.length));
}

export async function runBayesianOptimization<Input, Output>(
  promptCandidates: string[],
  dataset: DatasetRow<Input, Output>[],
  fn: OptimizableFunction<Input, Output>,
  scorer: (output: Output, expected: Output) => number,
  {
    model,
    numTrials,
    numFewShot,
    miniBatchSize,
    miniBatchFullEvalSteps,
  }: {
    model: LanguageModelV1;
    numTrials: number;
    numFewShot: number;
    miniBatchSize: number;
    miniBatchFullEvalSteps: number;
  }
) {
  const stats = new Map<string, { total: number; count: number }>();
  let best: {
    instruction: string;
    score: number;
    fewShotExamples: DatasetRow<Input, Output>[];
  } | null = null;
  const demosRegistry = new Map<string, DatasetRow<Input, Output>[]>();

  for (let t = 0; t < numTrials; t++) {
    console.log(`##### Trial ${t + 1} of ${numTrials} #####`);
    let instruction: string;

    if (t < promptCandidates.length) {
      instruction = promptCandidates[t];
    } else {
      let maxUcb = -Infinity;
      let bestInstr = promptCandidates[0];
      const logT = Math.log(t + 1);
      for (const cand of promptCandidates) {
        const stat = stats.get(cand);
        if (!stat) {
          bestInstr = cand;
          break;
        }
        const mean = stat.total / stat.count;
        const ucb = mean + Math.sqrt((2 * logT) / stat.count);
        if (ucb > maxUcb) {
          maxUcb = ucb;
          bestInstr = cand;
        }
      }
      instruction = bestInstr;
    }

    // Sample demos and mini-batch once per trial
    let demos = demosRegistry.get(instruction);
    if (!demos) {
      demos = selectRandomRowsFromDataset(dataset, numFewShot);
      demosRegistry.set(instruction, demos);
    }
    const miniBatch = selectRandomRowsFromDataset(dataset, miniBatchSize);
    const { score } = await evaluate(instruction, demos, miniBatch, fn, scorer);

    const current = stats.get(instruction) || { total: 0, count: 0 };
    current.total += score;
    current.count += 1;
    stats.set(instruction, current);

    if (!best || score > best.score)
      best = { instruction, score, fewShotExamples: demos };

    // Full-dataset evaluation every `miniBatchFullEvalSteps` trials
    if ((t + 1) % miniBatchFullEvalSteps === 0) {
      const fullScore = await evaluate(instruction, demos, dataset, fn, scorer);
      if (fullScore.score > (best?.score ?? 0)) {
        best = {
          instruction: fullScore.instruction,
          score: fullScore.score,
          fewShotExamples: fullScore.fewShotExamples,
        };
      }
    }
  }

  return best;
}
