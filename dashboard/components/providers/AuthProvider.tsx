'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getSupabaseClient } from '@/lib/supabase';
import { useAuthStore } from '@/lib/auth-store';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { setUser, setApiKeys } = useAuthStore();

  useEffect(() => {
    const supabase = getSupabaseClient();
    if (!supabase) return;

    // Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
      
      // Load API keys from localStorage
      const storedKeys = {
        openai: localStorage.getItem('nexify_openai_key') || undefined,
        anthropic: localStorage.getItem('nexify_anthropic_key') || undefined,
      };
      setApiKeys(storedKeys);
    });    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);

      if (!session) {
        // Clear keys on logout
        localStorage.removeItem('nexify_openai_key');
        localStorage.removeItem('nexify_anthropic_key');
        setApiKeys({ openai: undefined, anthropic: undefined });
      }
    });

    return () => subscription.unsubscribe();
  }, [setUser, setApiKeys, router]);

  return <>{children}</>;
}
