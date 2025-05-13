import { streamObject } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import { writeFileSync } from "node:fs";
import progress from "progress";

const NUM_SYNTHETIC_EXAMPLES = 100;
const prompt = `You are a synthetic data generator.

Return **exactly ${NUM_SYNTHETIC_EXAMPLES}** JSON objects (as a single JSON array) where each element follows this schema:

{
  "input": "<original e-mail thread proposing a meeting(s)>",
  "output": [{
    "subject": "<concise meeting subject>",
    "date": "<ISO-8601 date ‑ YYYY-MM-DD>",
    "time": "<24-hour time ‑ HH:MM>",
    "invitees": ["<Person 1>", "<Person 2>", ...]
  }, ...can have more elements]
}

Generation rules:
1. The **input** must look like a real e-mail thread (up to 50 sentences!, can be multiple emails) that proposes or reschedules meeting(s). Vary tone and context: casual catch-ups, formal reviews, official board mettings, quick syncs, project kick-offs, follow-ups, etc.
2. Dates should cover many months in the current and next calendar year. At least **10** e-mails should use relative expressions such as "next Friday" or "three weeks from today" – still resolve and place the concrete ISO date in the output.
3. Use diverse time expressions inside the e-mail (e.g. "7 a.m.", "15:30", "noon", "half past eight"), but always store **time** in output as strict 'HH:MM' 24-hour.
4. Vary the number of **invitees** between 1 and 5. All names in the invitees array must appear in the e-mail body.
5. Provide a short, descriptive **subject** ("Design Sync", "Budget Review Q3", etc.).
6. Mix stylistic elements: bullet lists, forwarded threads, quick notes from an assistant, mobile sign-offs, different salutations and signatures, etc., to maximise diversity.
7. Do **not** include any markdown, comments, or explanations—return only valid JSON.
8. Keep in mind that some of the emails will correct previously proposed meetings.
9. Some (but not all!) of the threads should be very messy, long and convoluted.
10. Some (but not all) of the threads should be extremely short and have no meetings at all (e.g. because they're cancelled in next email in the thread)`;

const stream = streamObject({
  prompt,
  model: openai("o3"),
  output: "array",
  schema: z.object({
    input: z.string(),
    output: z.array(
      z.object({
        subject: z.string(),
        date: z.string(),
        time: z.string(),
        invitees: z.array(z.string()),
      })
    ),
  }),
});

console.log(`Generating ${NUM_SYNTHETIC_EXAMPLES} synthetic examples...`);
const bar = new progress(
  "  generating [:bar] :rate/gen per sec, :percent, ETA: :etas",
  {
    total: NUM_SYNTHETIC_EXAMPLES,
    width: 20,
  }
);

const data = [];

for await (const object of stream.elementStream) {
  data.push(object);
  bar.tick();
  writeFileSync("data.json", JSON.stringify(data, null, 2));
}
