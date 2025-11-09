'use client';

import { useState, useEffect } from 'react';
import Header from '@/components/Header';
import { useAuthStore } from '@/lib/auth-store';
import { Key, Save, Eye, EyeOff } from 'lucide-react';
import toast from 'react-hot-toast';

export default function SettingsPage() {
  const { apiKeys, setApiKeys } = useAuthStore();
  const [openaiKey, setOpenaiKey] = useState('');
  const [anthropicKey, setAnthropicKey] = useState('');
  const [showOpenAI, setShowOpenAI] = useState(false);
  const [showAnthropic, setShowAnthropic] = useState(false);

  useEffect(() => {
    setOpenaiKey(apiKeys.openai || '');
    setAnthropicKey(apiKeys.anthropic || '');
  }, [apiKeys]);

  const handleSave = () => {
    // Save to localStorage
    if (openaiKey) {
      localStorage.setItem('nexify_openai_key', openaiKey);
    } else {
      localStorage.removeItem('nexify_openai_key');
    }
    
    if (anthropicKey) {
      localStorage.setItem('nexify_anthropic_key', anthropicKey);
    } else {
      localStorage.removeItem('nexify_anthropic_key');
    }

    // Update store
    setApiKeys({
      openai: openaiKey || undefined,
      anthropic: anthropicKey || undefined,
    });

    toast.success('API-Schlüssel erfolgreich gespeichert!');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black">
      <Header />

      <div className="container mx-auto px-4 py-12">
        <div className="max-w-3xl mx-auto">
          <div className="mb-8">
            <h1 className="text-4xl font-bold gradient-text mb-2">Einstellungen</h1>
            <p className="text-gray-400">Konfigurieren Sie Ihre API-Schlüssel</p>
          </div>

          <div className="bg-gray-900/50 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-gray-800">
            <div className="space-y-8">
              {/* OpenAI API Key */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Key className="w-5 h-5 text-purple-400" />
                  <label className="text-lg font-semibold text-white">
                    OpenAI API-Schlüssel
                  </label>
                </div>
                <p className="text-sm text-gray-400 mb-3">
                  Benötigt für GPT-4, GPT-3.5 und andere OpenAI-Modelle
                </p>
                <div className="relative">
                  <input
                    type={showOpenAI ? 'text' : 'password'}
                    value={openaiKey}
                    onChange={(e) => setOpenaiKey(e.target.value)}
                    className="w-full px-4 py-3 pr-12 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                    placeholder="sk-..."
                  />
                  <button
                    type="button"
                    onClick={() => setShowOpenAI(!showOpenAI)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    {showOpenAI ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Erhalten Sie Ihren Schlüssel bei{' '}
                  <a
                    href="https://platform.openai.com/api-keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-400 hover:underline"
                  >
                    platform.openai.com
                  </a>
                </p>
              </div>

              {/* Anthropic API Key */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Key className="w-5 h-5 text-blue-400" />
                  <label className="text-lg font-semibold text-white">
                    Anthropic API-Schlüssel
                  </label>
                </div>
                <p className="text-sm text-gray-400 mb-3">
                  Benötigt für Claude 3 Modelle (Optional)
                </p>
                <div className="relative">
                  <input
                    type={showAnthropic ? 'text' : 'password'}
                    value={anthropicKey}
                    onChange={(e) => setAnthropicKey(e.target.value)}
                    className="w-full px-4 py-3 pr-12 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                    placeholder="sk-ant-..."
                  />
                  <button
                    type="button"
                    onClick={() => setShowAnthropic(!showAnthropic)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    {showAnthropic ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                  Erhalten Sie Ihren Schlüssel bei{' '}
                  <a
                    href="https://console.anthropic.com/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:underline"
                  >
                    console.anthropic.com
                  </a>
                </p>
              </div>

              {/* Save Button */}
              <div className="pt-4">
                <button
                  onClick={handleSave}
                  className="w-full py-3 px-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold rounded-lg transition-all duration-200 flex items-center justify-center gap-2"
                >
                  <Save className="w-5 h-5" />
                  <span>Schlüssel speichern</span>
                </button>
              </div>

              {/* Info Box */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
                <p className="text-sm text-gray-300 mb-2">
                  <strong className="text-white">ℹ️ Hinweis:</strong>
                </p>
                <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
                  <li>Ihre Schlüssel werden nur lokal in Ihrem Browser gespeichert</li>
                  <li>Sie werden niemals an unsere Server übertragen</li>
                  <li>Mindestens ein Schlüssel wird benötigt, um den Chat zu nutzen</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
