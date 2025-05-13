import { readFileSync } from "fs";
import {
  evaluate,
  proposeInstructionCandidates,
  runBayesianOptimization,
  bootstrapFewShotExamples,
} from "./optimizer";
import type { DatasetRow } from "./types";
import { openai } from "@ai-sdk/openai";
import { extractMeetingsFromEmail } from "./extractMeetingsFromEmail";
import { arrayOfObjectsScorer } from "./scorers";

type Input = string;
type Output = {
  subject: string;
  date: string;
  time: string;
  invitees: string[];
}[];

const optimizationModel = openai("o3");
const numBootstrappedExamples = 3;
const numCandidates = 5; // Number of candidates to generate per trial
const numSamples = 10; // Number of samples to use for data summary

// This is the prompt that we'll be optimizing
const baseSystemPrompt = `Extract all meetings from the following email`;
const dataset: DatasetRow<Input, Output>[] = JSON.parse(
  readFileSync("data.json", "utf8")
);

// 1. Evaluate the base performance score (dumb instructions, no few-shot examples)
const { score } = await evaluate(
  baseSystemPrompt,
  [],
  dataset,
  extractMeetingsFromEmail,
  arrayOfObjectsScorer
);
console.log("Base performance score:", score);

// 2. Bootstrap few-shot examples - find {numBootstrappedExamples} examples that giving us correct results already
const fewShotExamples = await bootstrapFewShotExamples(
  baseSystemPrompt,
  dataset,
  numBootstrappedExamples,
  extractMeetingsFromEmail,
  arrayOfObjectsScorer
);
console.log("Found bootstrapped few-shot examples:", fewShotExamples);

// 3. Propose instruction candidates - generate {numCandidates} variations of the prompt
const candidates: string[] = await proposeInstructionCandidates(
  baseSystemPrompt,
  dataset,
  numCandidates,
  numSamples,
  optimizationModel
);

// 4. Run Bayesian optimization - find the best combination of instruction and few-shot examples using exploration and exploitation
const bestResult = await runBayesianOptimization(
  candidates,
  dataset,
  extractMeetingsFromEmail,
  arrayOfObjectsScorer,
  {
    model: optimizationModel,
    numTrials: 10,
    numFewShot: 3,
    miniBatchSize: 10,
    miniBatchFullEvalSteps: 10,
  }
);

console.log("Best instruction:", bestResult);
