// Program to run MIPROv2 optimization. It will
import { generateObject, generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { google } from "@ai-sdk/google";
import { z } from "zod";
import { readFileSync } from "node:fs";

interface DatasetRow {
  input: string;
  output: string;
}

const smartModel = openai("o3");
const model = google("gemini-1.5-flash");
const numTrials = 10;
const numCandidates = 5; // Number of candidates to generate per trial
const numSamples = 10; // Number of samples to use for data summary
const numLabeledExamples = 4; // Number of labeled examples to use for evaluation

const dataset: DatasetRow[] = JSON.parse(
  readFileSync("data/mipro_v2_dataset.json", "utf8")
);

// This is the prompt that we'll be optimizing
const baseSystemPrompt = ``;

async function generateDataSummary(dataset: DatasetRow[], sampleSize: number) {
  const sample = dataset.slice(0, sampleSize);
  const systemPrompt = `Analyze the following dataset examples and provide a summary of key patterns, input-output relationships, and any specific challenges the data presents. Focus on what makes a good answer and what patterns should be followed.`;
  const dataStr = `${sample.map((row) => JSON.stringify(row)).join("\n")}`;

  return (
    await generateText({
      system: systemPrompt,
      model: smartModel,
      prompt: dataStr,
    })
  ).text;
}

async function proposeInstructionCandidates(
  basePrompt: string,
  dataset: DatasetRow[],
  numCandidates: number,
  numSamples: number
) {
  const dataContext = await generateDataSummary(dataset, numSamples);
  const prompt = `Create ${numCandidates} high-quality instruction for an AI model performing the task described below.
    
  ${dataContext ? `<data-context>\n${dataContext}\n</data-context>\n\n` : ""}
  
Your task is to craft a clear, effective variation of the following instruction that will help the AI model generate accurate outputs for this task.
  
The instruction should be detailed enough to guide the model but not overly prescriptive or restrictive. Focus on what makes a good response rather than listing exact steps.`;

  return await generateObject({
    system: prompt,
    model: model,
    prompt: basePrompt,
    output: "array",
    schema: z.string(),
  });
}

async function runBayesianOptimization(promptCandidates: string[]) {
  const trials = [];
  for (let i = 0; i < numTrials; i++) {
    const trial = await runTrial(
      promptCandidates,
      dataset,
      numCandidates,
      numSamples
    );
  }
}

function selectLabeledExamples(
  dataset: DatasetRow[],
  maxExamples: number
): DatasetRow[] {
  const shuffled = [...dataset].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, Math.min(maxExamples, shuffled.length));
}

const labeledExamples = selectLabeledExamples(dataset, numLabeledExamples);
const candidates = await proposeInstructionCandidates(
  baseSystemPrompt,
  dataset,
  numCandidates,
  numSamples
);

console.log("Randomly selected labeled examples:", labeledExamples);
