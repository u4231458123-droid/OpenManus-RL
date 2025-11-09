import { OpenAIService } from './openai-service';
import { AnthropicService } from './anthropic-service';

export type AIProvider = 'openai' | 'anthropic';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export class AIManager {
  private openai: OpenAIService;
  private anthropic: AnthropicService;
  private defaultProvider: AIProvider = 'openai';

  constructor(openaiKey: string | null, anthropicKey: string | null) {
    this.openai = new OpenAIService(openaiKey);
    this.anthropic = new AnthropicService(anthropicKey);

    // Set default to whichever is available
    if (!this.openai.isAvailable() && this.anthropic.isAvailable()) {
      this.defaultProvider = 'anthropic';
    }
  }

  async chat(
    messages: ChatMessage[],
    provider?: AIProvider
  ): Promise<string> {
    const activeProvider = provider || this.defaultProvider;

    if (activeProvider === 'openai' && this.openai.isAvailable()) {
      return await this.openai.chat(messages);
    } else if (activeProvider === 'anthropic' && this.anthropic.isAvailable()) {
      // Filter out system messages for Anthropic and use as system prompt
      const systemMessages = messages.filter((m) => m.role === 'system');
      const systemPrompt = systemMessages.map((m) => m.content).join('\n');
      const chatMessages = messages.filter((m) => m.role !== 'system');

      return await this.anthropic.chat(
        chatMessages as Array<{ role: 'user' | 'assistant'; content: string }>,
        systemPrompt
      );
    }

    throw new Error(
      'No AI provider available. Please add your API keys in Settings.'
    );
  }

  async streamChat(
    messages: ChatMessage[],
    onChunk: (chunk: string) => void,
    provider?: AIProvider
  ): Promise<void> {
    const activeProvider = provider || this.defaultProvider;

    if (activeProvider === 'openai' && this.openai.isAvailable()) {
      return await this.openai.streamChat(messages, onChunk);
    } else if (activeProvider === 'anthropic' && this.anthropic.isAvailable()) {
      const systemMessages = messages.filter((m) => m.role === 'system');
      const systemPrompt = systemMessages.map((m) => m.content).join('\n');
      const chatMessages = messages.filter((m) => m.role !== 'system');

      return await this.anthropic.streamChat(
        chatMessages as Array<{ role: 'user' | 'assistant'; content: string }>,
        onChunk,
        systemPrompt
      );
    }

    throw new Error(
      'No AI provider available. Please add your API keys in Settings.'
    );
  }

  isAvailable(): boolean {
    return this.openai.isAvailable() || this.anthropic.isAvailable();
  }

  getAvailableProviders(): AIProvider[] {
    const providers: AIProvider[] = [];
    if (this.openai.isAvailable()) providers.push('openai');
    if (this.anthropic.isAvailable()) providers.push('anthropic');
    return providers;
  }
}
