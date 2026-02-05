export function extractTokenIds(tolistResult: (bigint | number)[][] | (bigint | number)[]): number[] {
  const flattened = Array.isArray(tolistResult[0]) ? (tolistResult as (bigint | number)[][]).flat() : (tolistResult as (bigint | number)[]);
  return flattened.map((v) => (typeof v === 'bigint' ? Number(v) : v));
}

export function copyEmbeddings(source: Float32Array, positions: number[], hiddenSize: number): Float32Array {
  const count = positions.length;
  const result = new Float32Array(count * hiddenSize);
  for (let i = 0; i < count; i++) {
    const sourceOffset = positions[i]! * hiddenSize;
    const destOffset = i * hiddenSize;
    for (let j = 0; j < hiddenSize; j++) {
      result[destOffset + j] = source[sourceOffset + j]!;
    }
  }
  return result;
}

export function sliceHiddenStates(source: Float32Array, startIdx: number, length: number, hiddenSize: number): Float32Array {
  const result = new Float32Array(length * hiddenSize);
  for (let i = 0; i < length; i++) {
    const sourceOffset = (startIdx + i) * hiddenSize;
    const destOffset = i * hiddenSize;
    for (let j = 0; j < hiddenSize; j++) {
      result[destOffset + j] = source[sourceOffset + j]!;
    }
  }
  return result;
}

export function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const expX = Math.exp(x);
  return expX / (1 + expX);
}

export function sigmoidArray(values: Float32Array): Float32Array {
  const result = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    result[i] = sigmoid(values[i]!);
  }
  return result;
}

export function getFirstOutput(result: Record<string, unknown>): Float32Array {
  const keys = Object.keys(result);
  if (keys.length === 0) {
    throw new Error('ONNX result has no outputs');
  }
  const tensor = result[keys[0]!] as { data: Float32Array };
  return tensor.data;
}

export function softmax(values: Float32Array): Float32Array {
  let maxVal = -Infinity;
  for (const val of values) {
    if (val > maxVal) {
      maxVal = val;
    }
  }

  const result = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const expVal = Math.exp(values[i]! - maxVal);
    result[i] = expVal;
    sum += expVal;
  }

  for (let i = 0; i < values.length; i++) {
    result[i] = result[i]! / sum;
  }
  return result;
}
