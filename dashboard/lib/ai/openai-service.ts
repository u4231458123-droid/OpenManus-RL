import OpenAI from 'openai';

export class OpenAIService {
  private client: OpenAI | null = null;

  constructor(apiKey: string | null) {
    if (apiKey) {
      this.client = new OpenAI({
        apiKey,
        dangerouslyAllowBrowser: true, // Only for demo - use server-side in production
      });
    }
  }

  async chat(
    messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>,
    model: string = 'gpt-4-turbo-preview'
  ): Promise<string> {
    if (!this.client) {
      throw new Error('OpenAI client not initialized. Please add your API key in Settings.');
    }

    try {
      const completion = await this.client.chat.completions.create({
        model,
        messages,
        temperature: 0.7,
        max_tokens: 2000,
      });

      return completion.choices[0]?.message?.content || 'No response generated';
    } catch (error: any) {
      console.error('OpenAI API error:', error);
      throw new Error(`OpenAI Error: ${error.message}`);
    }
  }

  async streamChat(
    messages: Array<{ role: 'user' | 'assistant' | 'system'; content: string }>,
    onChunk: (chunk: string) => void,
    model: string = 'gpt-4-turbo-preview'
  ): Promise<void> {
    if (!this.client) {
      throw new Error('OpenAI client not initialized. Please add your API key in Settings.');
    }

    try {
      const stream = await this.client.chat.completions.create({
        model,
        messages,
        temperature: 0.7,
        max_tokens: 2000,
        stream: true,
      });

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || '';
        if (content) {
          onChunk(content);
        }
      }
    } catch (error: any) {
      console.error('OpenAI streaming error:', error);
      throw new Error(`OpenAI Streaming Error: ${error.message}`);
    }
  }

  isAvailable(): boolean {
    return this.client !== null;
  }
}
