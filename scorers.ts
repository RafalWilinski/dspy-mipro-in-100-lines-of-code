export function arrayOfObjectsScorer(
  output: Record<string, any>[],
  expected: Record<string, any>[]
) {
  if (expected.length === 0) return 0;
  if (output.length === 0) return 0;

  const fieldScore = (
    out: Record<string, any>,
    exp: Record<string, any>
  ): number => {
    const keys = Object.keys(exp);
    let hits = 0;

    for (const key of keys) {
      const expVal = exp[key];
      const outVal = out[key];

      if (Array.isArray(expVal) && Array.isArray(outVal)) {
        if (expVal.length === 0) {
          hits += 1;
          continue;
        }
        const common = expVal.filter((v) => outVal.includes(v)).length;
        hits += common / expVal.length;
      } else if (expVal === outVal) {
        hits += 1;
      }
    }

    return hits / keys.length;
  };

  let cumulative = 0;
  for (const exp of expected) {
    let bestMatch = 0;
    for (const out of output) {
      bestMatch = Math.max(bestMatch, fieldScore(out, exp));
      if (bestMatch === 1) break; // perfect match found
    }
    cumulative += bestMatch;
  }

  return cumulative / expected.length;
}
