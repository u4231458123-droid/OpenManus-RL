import Anthropic from '@anthropic-ai/sdk';

export class AnthropicService {
  private client: Anthropic | null = null;

  constructor(apiKey: string | null) {
    if (apiKey) {
      this.client = new Anthropic({
        apiKey,
        dangerouslyAllowBrowser: true, // Only for demo - use server-side in production
      });
    }
  }

  async chat(
    messages: Array<{ role: 'user' | 'assistant'; content: string }>,
    systemPrompt?: string,
    model: string = 'claude-3-sonnet-20240229'
  ): Promise<string> {
    if (!this.client) {
      throw new Error('Anthropic client not initialized. Please add your API key in Settings.');
    }

    try {
      const response = await this.client.messages.create({
        model,
        max_tokens: 2000,
        system: systemPrompt || 'You are a helpful AI assistant.',
        messages,
      });

      const content = response.content[0];
      if (content.type === 'text') {
        return content.text;
      }

      return 'No response generated';
    } catch (error: any) {
      console.error('Anthropic API error:', error);
      throw new Error(`Anthropic Error: ${error.message}`);
    }
  }

  async streamChat(
    messages: Array<{ role: 'user' | 'assistant'; content: string }>,
    onChunk: (chunk: string) => void,
    systemPrompt?: string,
    model: string = 'claude-3-sonnet-20240229'
  ): Promise<void> {
    if (!this.client) {
      throw new Error('Anthropic client not initialized. Please add your API key in Settings.');
    }

    try {
      const stream = await this.client.messages.stream({
        model,
        max_tokens: 2000,
        system: systemPrompt || 'You are a helpful AI assistant.',
        messages,
      });

      for await (const chunk of stream) {
        if (
          chunk.type === 'content_block_delta' &&
          chunk.delta.type === 'text_delta'
        ) {
          onChunk(chunk.delta.text);
        }
      }
    } catch (error: any) {
      console.error('Anthropic streaming error:', error);
      throw new Error(`Anthropic Streaming Error: ${error.message}`);
    }
  }

  isAvailable(): boolean {
    return this.client !== null;
  }
}
