import { create } from 'zustand';
import { User } from '@supabase/supabase-js';

interface AuthState {
  user: User | null;
  apiKeys: {
    openai: string | null;
    anthropic: string | null;
  };
  setUser: (user: User | null) => void;
  setApiKeys: (keys: { openai?: string; anthropic?: string }) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  apiKeys: {
    openai: null,
    anthropic: null,
  },
  setUser: (user) => set({ user }),
  setApiKeys: (keys) =>
    set((state) => ({
      apiKeys: {
        ...state.apiKeys,
        ...keys,
      },
    })),
  logout: () =>
    set({
      user: null,
      apiKeys: { openai: null, anthropic: null },
    }),
}));
