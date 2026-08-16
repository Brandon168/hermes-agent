import { createGateway } from '@ai-sdk/gateway';
import { generateText, stepCountIs } from 'ai';

const apiKey = process.env.AI_GATEWAY_API_KEY;
if (!apiKey) {
  throw new Error('AI_GATEWAY_API_KEY is required');
}

let wire;
const tracedFetch = async (url, init = {}) => {
  const response = await fetch(url, init);
  wire = {
    url: String(url),
    requestHeaders: Object.fromEntries(
      [...new Headers(init.headers).entries()].filter(
        ([key]) => key.toLowerCase() !== 'authorization',
      ),
    ),
    requestBody: JSON.parse(String(init.body)),
    status: response.status,
    responseHeaders: Object.fromEntries(response.headers.entries()),
    responseBody: await response.clone().text(),
  };
  return response;
};

const gateway = createGateway({ apiKey, fetch: tracedFetch });
const model = gateway('openai/gpt-5.6-luna');
const result = await generateText({
  model,
  prompt: 'Search for one official Vercel page about AI Gateway and answer with its title.',
  tools: {
    web_search: gateway.tools.exaSearch({
      type: 'fast',
      numResults: 1,
      contents: { text: { maxCharacters: 1200 } },
    }),
  },
  toolChoice: 'auto',
  providerOptions: {
    gateway: { zeroDataRetention: true },
  },
  reasoning: 'low',
  stopWhen: stepCountIs(4),
});

console.log(JSON.stringify({
  oracle: {
    ai: '7.0.66',
    '@ai-sdk/gateway': '4.0.52',
    '@ai-sdk/provider': '4.0.7',
  },
  wire,
  result: {
    text: result.text,
    finishReason: result.finishReason,
    usage: result.usage,
    providerMetadata: result.providerMetadata,
    steps: result.steps.map((step) => ({
      finishReason: step.finishReason,
      toolCalls: step.toolCalls,
      toolResults: step.toolResults,
      sources: step.sources,
      usage: step.usage,
      providerMetadata: step.providerMetadata,
    })),
  },
}, null, 2));
