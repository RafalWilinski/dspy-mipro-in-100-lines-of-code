import { openai } from "@ai-sdk/openai";
import { generateObject } from "ai";
import { z } from "zod";
import type { DatasetRow, OptimizableFunction } from "./types";
import { google } from "@ai-sdk/google";

const meetingSchema = z.object({
  subject: z.string(),
  date: z.string(),
  time: z.string(),
  invitees: z.array(z.string()),
});

type Input = string;
type Output = {
  subject: string;
  date: string;
  time: string;
  invitees: string[];
}[];

export const extractMeetingsFromEmail: OptimizableFunction<
  Input,
  Output
> = async (
  email: string,
  fewShotExamples: DatasetRow<string, Output>[],
  instructions: string = "Extract all meetings from the following email"
) => {
  const prompt = `${instructions}

Examples:

<examples>
${fewShotExamples
  .map(
    (example) => `
<example>
<email>
${example.input}
</email>
<output>
${example.output}
</output>
`
  )
  .join("\n")}
</examples>

<email>
${email}
</email>
  `;

  const response = await generateObject({
    // model: openai("gpt-3.5-turbo"), // Quite dumb model
    model: google("gemini-2.0-flash"),
    prompt,
    output: "array",
    schema: meetingSchema,
  });

  return response.object;
};
