/**
 * Clearly-marked demo inputs for development convenience.
 *
 * These are INPUTS ONLY — nothing here is a prediction. Results always come
 * from the backend model; when the backend is unavailable the UI shows an
 * error instead of pretending these examples were analyzed.
 */

export interface DemoExample {
  id: string;
  label: string;
  context: string;
  comment: string;
}

export const demoExamples: DemoExample[] = [
  {
    id: "demo-immigration",
    label: "Immigration context",
    context: "Why are you talking about immigrants?",
    comment: "Those people are disgusting and should leave.",
  },
  {
    id: "demo-neutral",
    label: "Neutral chat",
    context: "The match yesterday was wild.",
    comment: "I still can't believe that final goal.",
  },
  {
    id: "demo-insult",
    label: "Direct insult",
    context: "Nobody asked for your opinion.",
    comment: "You are such an idiot, just shut up already.",
  },
];
