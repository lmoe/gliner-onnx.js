# gliner-onnx

GLiNER ONNX runtime for JavaScript/TypeScript. Runs GLiNER models without PyTorch.

This library is experimental. The API may change between versions.

## Features

- GLiNER2: Zero-shot NER and text classification
- GLiNER1: Zero-shot NER (fork of [Knowledgator/GLiNER.js](https://github.com/Knowledgator/GLiNER.js), refactored with updated dependencies)

## Installation

```bash
npm install @lmoe/gliner-onnx
```

## GLiNER2

GLiNER2 supports both NER and classification.

### NER

```typescript
import { GLiNER2ONNXRuntime } from '@lmoe/gliner-onnx';

const model = await GLiNER2ONNXRuntime.fromPretrained('lmo3/gliner2-large-v1-onnx');

const entities = await model.extractEntities(
  'John works at Google in Seattle',
  ['person', 'organization', 'location']
);
// [
//   { text: 'John', label: 'person', start: 0, end: 4, score: 0.98 },
//   { text: 'Google', label: 'organization', start: 14, end: 20, score: 0.97 },
//   { text: 'Seattle', label: 'location', start: 24, end: 31, score: 0.96 }
// ]
```

### Classification

```typescript
import { GLiNER2ONNXRuntime } from '@lmoe/gliner-onnx';

const model = await GLiNER2ONNXRuntime.fromPretrained('lmo3/gliner2-multi-v1-onnx');

// Single-label classification
const result = await model.classify(
  'Buy milk from the store',
  ['shopping', 'work', 'entertainment']
);
// { shopping: 0.95 }

// Multi-label classification
const multi = await model.classify(
  'Buy milk and finish the report',
  ['shopping', 'work', 'entertainment'],
  { multiLabel: true, threshold: 0.3 }
);
// { shopping: 0.85, work: 0.72 }
```

## GLiNER1

GLiNER1 supports NER only. Use GLiNER2 if you need classification.

The GLiNER1 implementation is a fork of [Knowledgator/GLiNER.js](https://github.com/Knowledgator/GLiNER.js), refactored and updated.

### NER

```typescript
import { GLiNER1ONNXRuntime } from '@lmoe/gliner-onnx';

const model = await GLiNER1ONNXRuntime.fromPretrained('onnx-community/gliner_small-v2.1');

const entities = await model.extractEntities(
  'John works at Google in Seattle',
  ['person', 'organization', 'location']
);
// [
//   { text: 'John', label: 'person', start: 0, end: 4, score: 0.98 },
//   { text: 'Google', label: 'organization', start: 14, end: 20, score: 0.97 },
//   { text: 'Seattle', label: 'location', start: 24, end: 31, score: 0.96 }
// ]
```

### Batch Processing

```typescript
const results = await model.extractEntitiesBatch(
  ['John works at Google', 'Mary lives in Paris'],
  ['person', 'organization', 'location']
);
```

## CUDA

To use CUDA for GPU acceleration:

```typescript
import { GLiNER2ONNXRuntime } from '@lmoe/gliner-onnx';

const model = await GLiNER2ONNXRuntime.fromPretrained('lmo3/gliner2-large-v1-onnx', {
  executionProviders: ['cuda', 'cpu'],
});
```

The same option works for GLiNER1:

```typescript
import { GLiNER1ONNXRuntime } from '@lmoe/gliner-onnx';

const model = await GLiNER1ONNXRuntime.fromPretrained('onnx-community/gliner_small-v2.1', {
  executionProviders: ['cuda', 'cpu'],
});
```

## Precision

Both fp32 and fp16 models are supported. To use fp16:

```typescript
const model = await GLiNER2ONNXRuntime.fromPretrained('lmo3/gliner2-large-v1-onnx', {
  precision: 'fp16',
});
```

## Models

### GLiNER2

GLiNER2 models need to be exported to ONNX format. Pre-exported models:

- [lmo3/gliner2-large-v1-onnx](https://huggingface.co/lmo3/gliner2-large-v1-onnx)
- [lmo3/gliner2-multi-v1-onnx](https://huggingface.co/lmo3/gliner2-multi-v1-onnx)

To export your own models, see the Python exporter: [lmoe/gliner2-onnx](https://github.com/lmoe/gliner2-onnx)

### GLiNER1

GLiNER1 models from [onnx-community](https://huggingface.co/onnx-community) work directly:

- [onnx-community/gliner_small-v2.1](https://huggingface.co/onnx-community/gliner_small-v2.1)
- [onnx-community/gliner_medium-v2.1](https://huggingface.co/onnx-community/gliner_medium-v2.1)
- [onnx-community/gliner_large-v2.1](https://huggingface.co/onnx-community/gliner_large-v2.1)

## Credits

- [Knowledgator/GLiNER.js](https://github.com/Knowledgator/GLiNER.js): GLiNER1 code is based on this project
- [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1): Pre-trained GLiNER2 models

## License

MIT
