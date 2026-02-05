import type { Entity } from '../types.js';
import type { ClassificationResult, WordOffset } from './types.js';
import { sigmoid, sigmoidArray, softmax } from './utils.js';

export interface SpanIndices {
  spanStart: number[];
  spanEnd: number[];
  numSpans: number;
}

/** Generate all possible spans up to maxWidth. */
export function generateSpans(seqLen: number, maxWidth: number): SpanIndices {
  const spanStart: number[] = [];
  const spanEnd: number[] = [];

  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < maxWidth; j++) {
      if (i + j < seqLen) {
        spanStart.push(i);
        spanEnd.push(i + j);
      } else {
        spanStart.push(0);
        spanEnd.push(0);
      }
    }
  }

  return { spanStart, spanEnd, numSpans: spanStart.length };
}

/** Compute dot product scores between spans and labels with sigmoid activation. */
export function computeDotProductScores(
  spanRep: Float32Array,
  labelRep: Float32Array,
  spanCount: number,
  labelCount: number,
  hiddenSize: number
): Float32Array {
  const scores = new Float32Array(spanCount * labelCount);

  for (let spanIdx = 0; spanIdx < spanCount; spanIdx++) {
    for (let labelIdx = 0; labelIdx < labelCount; labelIdx++) {
      let dotProduct = 0;
      for (let hiddenIdx = 0; hiddenIdx < hiddenSize; hiddenIdx++) {
        dotProduct += spanRep[spanIdx * hiddenSize + hiddenIdx]! * labelRep[labelIdx * hiddenSize + hiddenIdx]!;
      }
      scores[spanIdx * labelCount + labelIdx] = sigmoid(dotProduct);
    }
  }

  return scores;
}

export interface NerScoreData {
  scores: Float32Array;
  wordSpanStart: number[];
  wordSpanEnd: number[];
  spanCount: number;
}

/** Decode entities from NER scores. */
export function decodeEntities(
  scoreData: NerScoreData,
  wordCount: number,
  labels: readonly string[],
  wordOffsets: WordOffset[],
  text: string,
  threshold: number
): Entity[] {
  const { scores, wordSpanStart, wordSpanEnd, spanCount } = scoreData;
  const labelCount = labels.length;
  const entities: Entity[] = [];

  for (let spanIdx = 0; spanIdx < spanCount; spanIdx++) {
    const startWord = wordSpanStart[spanIdx]!;
    const endWord = wordSpanEnd[spanIdx]!;

    if (startWord >= wordCount || endWord >= wordCount) {
      continue;
    }

    for (let labelIdx = 0; labelIdx < labelCount; labelIdx++) {
      const score = scores[spanIdx * labelCount + labelIdx]!;

      if (score >= threshold) {
        const startOffset = wordOffsets[startWord]!;
        const endOffset = wordOffsets[endWord]!;

        entities.push({
          text: text.slice(startOffset[0], endOffset[1]),
          label: labels[labelIdx]!,
          start: startOffset[0],
          end: endOffset[1],
          score,
        });
      }
    }
  }

  return deduplicateEntities(entities);
}

/** Remove overlapping entities, keeping highest scoring ones. */
function deduplicateEntities(entities: Entity[]): Entity[] {
  if (entities.length === 0) {
    return [];
  }

  const sorted = [...entities].sort((a, b) => b.score - a.score);
  const kept: Entity[] = [];

  for (const entity of sorted) {
    const hasOverlap = kept.some((existing) => entity.label === existing.label && entity.start < existing.end && entity.end > existing.start);
    if (!hasOverlap) {
      kept.push(entity);
    }
  }

  return kept;
}

/** Decode multi-label classification results. */
export function decodeMultiLabel(logits: Float32Array, labels: readonly string[], threshold: number): ClassificationResult {
  const probabilities = sigmoidArray(logits);
  const results: ClassificationResult = {};

  for (let i = 0; i < labels.length; i++) {
    if (probabilities[i]! >= threshold) {
      results[labels[i]!] = probabilities[i]!;
    }
  }
  return results;
}

/** Decode single-label classification result. */
export function decodeSingleLabel(logits: Float32Array, labels: readonly string[]): ClassificationResult {
  const probabilities = softmax(logits);

  let bestIdx = 0;
  let bestProb = probabilities[0]!;
  for (let i = 1; i < labels.length; i++) {
    if (probabilities[i]! > bestProb) {
      bestIdx = i;
      bestProb = probabilities[i]!;
    }
  }

  return { [labels[bestIdx]!]: bestProb };
}
