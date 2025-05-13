export interface DatasetRow<Input, Output> {
  input: Input;
  output: Output;
}

export type OptimizableFunction<Input, Output> = (
  input: Input,
  fewShotExamples: DatasetRow<Input, Output>[],
  instructions: string
) => Promise<Output>;
