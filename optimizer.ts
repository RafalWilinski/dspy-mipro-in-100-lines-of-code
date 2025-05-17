import { generateObject, generateText, type LanguageModelV1 } from "ai";
import { z } from "zod";
import type { DatasetRow, OptimizableFunction } from "./types";

const INSTRUCTION_TIPS = [
  "Be very specific and detailed in your instructions.",
  "Focus on step-by-step reasoning in your instructions.",
  "Provide clear constraints and guidelines in your instructions.",
  "Keep your instructions concise and to the point.",
  "Emphasize accuracy and precision in your instructions.",
  "Include examples of good outputs in your instructions.",
  "Focus on handling edge cases in your instructions.",
  "Explicitly outline the reasoning process in your instructions.",
];

/**
 * Summarize a random sample of dataset rows so the language model can quickly
 * grasp key input–output patterns and common pitfalls.
 */
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

/**
 * Ask an LLM to propose multiple alternative task instructions, seeded with a
 * data-driven context summary and the original prompt.
 */
export async function proposeInstructionCandidates<Input, Output>(
  prompt: string,
  dataset: DatasetRow<Input, Output>[],
  numCandidates: number,
  numSamples: number,
  model: LanguageModelV1,
  highScoringTraces: DatasetRow<Input, Output>[] = []
): Promise<string[]> {
  console.log(`Generating ${numCandidates} instruction candidates`);
  const dataContext = await generateDataSummary(dataset, numSamples, model);

  let traceContext = "";
  if (highScoringTraces.length > 0) {
    const serialized = highScoringTraces
      .slice(0, 5) // cap to avoid giant prompts
      .map((row) => JSON.stringify(row))
      .join("\n");
    traceContext = `<high-scoring-traces>\n${serialized}\n</high-scoring-traces>`;
  }
  const system = `Create ${numCandidates} high-quality instruction for an AI model performing the task described below.
    
  ${dataContext ? `<data-context>\n${dataContext}\n</data-context>\n\n` : ""}
  ${traceContext ? `${traceContext}\n\n` : ""}
  Your task is to craft a clear, effective variation of the following instruction that will help the AI model generate accurate outputs for this task.
  
  The instruction should be detailed enough to guide the model but not overly prescriptive or restrictive. Focus on what makes a good response rather than listing exact steps.`;

  const result = await generateObject({
    system,
    model,
    prompt,
    output: "array",
    schema: z.object({ newPrompt: z.string() }),
  });
  return result.object.map(
    (o) =>
      `${o.newPrompt}\n\nTip: ${
        INSTRUCTION_TIPS[Math.floor(Math.random() * INSTRUCTION_TIPS.length)]
      }`
  );
}

/**
 * Run the target function on evaluation rows using the provided instruction
 * and few-shot examples, then return the average metric score.
 */
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

/**
 * Randomly shuffle the dataset and return up to `maxExamples` rows.
 */
export function selectRandomRowsFromDataset<Input, Output>(
  dataset: DatasetRow<Input, Output>[],
  maxExamples: number
): DatasetRow<Input, Output>[] {
  const shuffled = [...dataset].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(maxExamples, shuffled.length));
}

/**
 * Create `numDemoCandidates` demo pools of size `numFewShot`.
 * For each candidate we randomly iterate through dataset rows, run the teacher
 * program (fn + basePrompt) and keep only rows whose generated output passes
 * the metric (score === 1). If we run out of rows before filling the pool, we
 * back-fill with random dataset rows so we always return fully-sized pools.
 */
async function generateDemoCandidates<Input, Output>(
  dataset: DatasetRow<Input, Output>[],
  numDemoCandidates: number,
  numFewShot: number,
  basePrompt: string,
  fn: OptimizableFunction<Input, Output>,
  scorer: (output: Output, expected: Output) => number
): Promise<DatasetRow<Input, Output>[][]> {
  const candidates: DatasetRow<Input, Output>[][] = [];

  for (let c = 0; c < numDemoCandidates; c++) {
    const demos: DatasetRow<Input, Output>[] = [];
    const shuffled = [...dataset].sort(() => Math.random() - 0.5);

    for (const row of shuffled) {
      if (demos.length >= numFewShot) break;
      // Teacher attempt – generate output using base prompt (no few-shot)
      const output = await fn(row.input, [], basePrompt);
      if (scorer(output, row.output) === 1) {
        demos.push(row);
      }
    }

    // If not enough passing demos, top-up with random rows to maintain size
    if (demos.length < numFewShot) {
      const filler = selectRandomRowsFromDataset(
        dataset,
        numFewShot - demos.length
      );
      demos.push(...filler);
    }

    candidates.push(demos);
  }

  return candidates;
}

/**
 * Coordinate a bandit-style search over instruction × demo pools using an
 * upper-confidence-bound policy to uncover the best performing combo.
 */
export async function runBayesianOptimization<Input, Output>(
  promptCandidates: string[],
  dataset: DatasetRow<Input, Output>[],
  fn: OptimizableFunction<Input, Output>,
  scorer: (output: Output, expected: Output) => number,
  {
    model: _model,
    numTrials,
    numFewShot,
    numDemoCandidates,
    miniBatchSize,
    miniBatchFullEvalSteps,
  }: {
    model: LanguageModelV1;
    numTrials: number;
    numFewShot: number;
    numDemoCandidates: number;
    miniBatchSize: number;
    miniBatchFullEvalSteps: number;
  }
) {
  // Generate demo candidate pools using teacher-guided metric passing
  const teacherPrompt = promptCandidates[0];
  const demoCandidates = await generateDemoCandidates(
    dataset,
    numDemoCandidates,
    numFewShot,
    teacherPrompt,
    fn,
    scorer
  );

  // Create all instruction–demo combinations
  type Combo = {
    instruction: string;
    demos: DatasetRow<Input, Output>[];
    key: string;
  };
  const combos: Combo[] = [];
  const existingInstructions = new Set<string>();
  for (const instr of promptCandidates) {
    existingInstructions.add(instr);
    for (let i = 0; i < demoCandidates.length; i++) {
      const key = `${instr}::${i}`;
      combos.push({ instruction: instr, demos: demoCandidates[i], key });
    }
  }

  const stats = new Map<string, { total: number; count: number }>();
  let best: {
    instruction: string;
    score: number;
    fewShotExamples: DatasetRow<Input, Output>[];
  } | null = null;

  for (let t = 0; t < numTrials; t++) {
    console.log(`##### Trial ${t + 1} of ${numTrials} #####`);

    // Select combo using UCB policy
    let combo: Combo;
    if (t < combos.length) {
      combo = combos[t]; // warm-start: ensure each combo is tried once
    } else {
      const logT = Math.log(t + 1);
      let maxUcb = -Infinity;
      combo = combos[0];
      for (const c of combos) {
        const stat = stats.get(c.key);
        if (!stat) {
          combo = c;
          break;
        }
        const mean = stat.total / stat.count;
        const ucb = mean + Math.sqrt((2 * logT) / stat.count);
        if (ucb > maxUcb) {
          maxUcb = ucb;
          combo = c;
        }
      }
    }

    const miniBatch = selectRandomRowsFromDataset(dataset, miniBatchSize);
    const { score } = await evaluate(
      combo.instruction,
      combo.demos,
      miniBatch,
      fn,
      scorer
    );

    const current = stats.get(combo.key) || { total: 0, count: 0 };
    current.total += score;
    current.count += 1;
    stats.set(combo.key, current);

    if (!best || score > best.score) {
      best = {
        instruction: combo.instruction,
        score,
        fewShotExamples: combo.demos,
      };
    }

    // Full-dataset evaluation every `miniBatchFullEvalSteps` trials
    if ((t + 1) % miniBatchFullEvalSteps === 0) {
      const fullScore = await evaluate(
        combo.instruction,
        combo.demos,
        dataset,
        fn,
        scorer
      );
      if (fullScore.score > (best?.score ?? 0)) {
        best = {
          instruction: fullScore.instruction,
          score: fullScore.score,
          fewShotExamples: fullScore.fewShotExamples,
        };
      }

      // ---- NEW: iterative instruction generation ----
      // After each full evaluation step, generate additional instruction candidates
      const numNewCandidates = 3; // small to keep budget low
      const newCandidates = await proposeInstructionCandidates(
        best.instruction,
        dataset,
        numNewCandidates,
        miniBatchSize,
        _model,
        best.fewShotExamples ?? []
      );

      for (const newInstr of newCandidates) {
        if (existingInstructions.has(newInstr)) continue; // avoid duplicates
        existingInstructions.add(newInstr);
        for (let i = 0; i < demoCandidates.length; i++) {
          const key = `${newInstr}::${i}`;
          combos.push({ instruction: newInstr, demos: demoCandidates[i], key });
        }
      }
      // ----------------------------------------------
    }
  }

  return best;
}
