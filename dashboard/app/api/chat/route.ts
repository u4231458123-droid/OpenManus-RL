import { NextRequest, NextResponse } from 'next/server'

export const runtime = 'edge'
export const dynamic = 'force-dynamic'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatRequest {
  message: string
  history: ChatMessage[]
}

export async function POST(req: NextRequest) {
  try {
    const { message, history }: ChatRequest = await req.json()

    // TODO: Integrate with OpenManus Python backend
    // For now, return a simulated response

    // Simulate thinking process
    const thinking = analyzeQuery(message)

    // Simulate tool usage
    const toolsUsed = determineTools(message)

    // Generate response
    const response = await generateResponse(message, thinking, toolsUsed)

    return NextResponse.json({
      response,
      thinking,
      tools_used: toolsUsed,
      success: true
    })
  } catch (error) {
    console.error('Chat API error:', error)
    return NextResponse.json(
      { error: 'Failed to process message', success: false },
      { status: 500 }
    )
  }
}

function analyzeQuery(message: string): string {
  const lower = message.toLowerCase()

  if (lower.includes('such') || lower.includes('find') || lower.includes('recherch')) {
    return 'Analysiere Suchanfrage und wähle passende Tools...'
  }
  if (lower.includes('bild') || lower.includes('foto') || lower.includes('image')) {
    return 'Bildanalyse wird vorbereitet...'
  }
  if (lower.includes('code') || lower.includes('python') || lower.includes('programm')) {
    return 'Code-Generierung wird initialisiert...'
  }
  if (lower.includes('wetter') || lower.includes('weather')) {
    return 'Wetterdaten werden abgerufen...'
  }

  return 'Verarbeite Anfrage und plane Antwort...'
}

function determineTools(message: string): string[] {
  const lower = message.toLowerCase()
  const tools: string[] = []

  if (lower.includes('such') || lower.includes('google') || lower.includes('web')) {
    tools.push('google_search')
  }
  if (lower.includes('wikipedia') || lower.includes('wiki')) {
    tools.push('wikipedia_knowledge_searcher')
  }
  if (lower.includes('arxiv') || lower.includes('paper') || lower.includes('forschung')) {
    tools.push('arxiv_paper_searcher')
  }
  if (lower.includes('bild') || lower.includes('foto')) {
    tools.push('image_captioner', 'advanced_object_detector')
  }
  if (lower.includes('code') || lower.includes('python')) {
    tools.push('python_code_generator')
  }
  if (lower.includes('url') || lower.includes('website') || lower.includes('link')) {
    tools.push('url_text_extractor')
  }

  return tools
}

async function generateResponse(
  message: string,
  thinking: string,
  toolsUsed: string[]
): Promise<string> {
  // Simulated responses based on query type
  const lower = message.toLowerCase()

  if (lower.includes('hallo') || lower.includes('hi') || lower.includes('hey')) {
    return 'Hallo! Ich bin OpenManus AI, ein KI-Assistent mit Zugriff auf verschiedene Tools wie Web-Suche, Bildanalyse, Code-Generierung und mehr. Wie kann ich Ihnen helfen?'
  }

  if (toolsUsed.length > 0) {
    return `Ich habe Ihre Anfrage analysiert und würde folgende Tools verwenden: ${toolsUsed.join(', ')}.\n\nHinweis: Die volle Backend-Integration ist in Arbeit. Momentan ist dies eine Demo-Oberfläche.\n\nMöchten Sie mehr über die verfügbaren Funktionen erfahren?`
  }

  if (lower.includes('was kannst du')) {
    return `Ich kann Sie bei vielen Aufgaben unterstützen:\n\n🔍 **Web-Suche**: Aktuelle Informationen aus dem Internet\n📚 **Wissensabfrage**: Wikipedia, Arxiv-Papers\n🛠️ **Tool-Nutzung**: Python-Code, Bildanalyse, Text-Extraktion\n🧠 **Reasoning**: Multi-Turn Dialoge mit Planung\n📊 **Analytics**: Training-Metriken und Performance\n\nWas interessiert Sie am meisten?`
  }

  if (lower.includes('tool') || lower.includes('funktion')) {
    return `Aktuell verfügbare Tools:\n\n• Google Search - Web-Recherche\n• Wikipedia Knowledge Searcher\n• Arxiv Paper Searcher\n• Image Captioner - Bildbeschreibung\n• Advanced Object Detector\n• Python Code Generator\n• URL Text Extractor\n• Text Detector\n• Nature News Fetcher\n\nWelches Tool möchten Sie ausprobieren?`
  }

  return `Vielen Dank für Ihre Nachricht! Dies ist eine Demo der OpenManus AI Oberfläche.\n\nDie vollständige Integration mit dem Python-Backend (Reinforcement Learning Agent, Tool-Integration) ist in Entwicklung.\n\nMöchten Sie mehr über die geplanten Features erfahren oder das Analytics-Dashboard ansehen?`
}
