import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.NEXT_PUBLIC_GEMINI_API_KEY! });

export interface MultimodalPart {
  text?: string;
  inlineData?: {
    data: string;
    mimeType: string;
  };
}

export async function getEmbedding(input: string | MultimodalPart[]): Promise<number[] | null> {
  const multimodalModel = "gemini-embedding-2-preview";
  
  let parts: any[] = [];
  if (typeof input === 'string') {
    parts = [{ text: input }];
  } else {
    parts = input;
  }

  try {
    const response = await ai.models.embedContent({
      model: multimodalModel,
      contents: [{ parts }],
    });
    
    const values = response.embeddings?.[0]?.values;
    if (!values) {
      throw new Error("No embeddings returned from API");
    }
    return values;
  } catch (error) {
    console.error("Embedding error:", error);
    throw error;
  }
}

export async function getBatchEmbeddings(inputs: (string | MultimodalPart[])[]) {
  const multimodalModel = "gemini-embedding-2-preview";
  
  try {
    const response = await ai.models.embedContent({
      model: multimodalModel,
      contents: inputs.map(input => {
        const parts = typeof input === 'string' ? [{ text: input }] : input;
        return { parts };
      }),
    });
    
    if (!response.embeddings) {
      throw new Error("No embeddings returned from API");
    }
    return response.embeddings.map(e => e.values || null);
  } catch (error) {
    console.error("Batch embedding error:", error);
    throw error;
  }
}

export function cosineSimilarity(vecA: number[] | null | undefined, vecB: number[] | null | undefined) {
  if (!vecA || !vecB) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
