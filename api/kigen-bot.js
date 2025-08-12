// api/kigen-bot.js
import fs from "fs/promises";
import path from "path";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const { question } = req.body || {};
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Missing question" });
    }

    // 1) اقرأ قاعدة المعرفة
    const kbPath = path.join(process.cwd(), "kb", "kigen_knowledge.json");
    const raw = await (await fs.readFile(kbPath, "utf8"));
    const kb = JSON.parse(raw);

    // 2) تحضير الاتصال بـ OpenAI
    const headers = {
      "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    };

    // 3) احصل على Embedding للسؤال
    const qEmbRes = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST", headers,
      body: JSON.stringify({ model: "text-embedding-3-small", input: question })
    });
    const qEmbJson = await qEmbRes.json();
    const qEmb = qEmbJson?.data?.[0]?.embedding;
    if (!Array.isArray(qEmb)) {
      console.error("Embedding error:", qEmbJson);
      return res.status(500).json({ error: "Failed to embed question" });
    }

    // 4) احصل على Embeddings لكل مقطع (ساذج الآن؛ لاحقًا نخزّنها مسبقًا)
    const chunks = [];
    for (const chunk of kb) {
      const embRes = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST", headers,
        body: JSON.stringify({ model: "text-embedding-3-small", input: chunk.content })
      });
      const embJson = await embRes.json();
      const emb = embJson?.data?.[0]?.embedding;
      if (Array.isArray(emb)) chunks.push({ chunk, emb });
    }

    // 5) دالة تشابه كوني
    const cosine = (a, b) => {
      let dot = 0, na = 0, nb = 0;
      for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
      return dot / (Math.sqrt(na) * Math.sqrt(nb));
    };

    // 6) ترتيب المقاطع بالأعلى صلة
    const ranked = chunks.map(({ chunk, emb }) => ({
      chunk, score: cosine(qEmb, emb)
    })).sort((x, y) => y.score - x.score);

    const TOP_K = 3;
    const MIN_SCORE = 0.18; // عدّليها حسب الحاجة
    const top = ranked.slice(0, TOP_K).filter(r => r.score >= MIN_SCORE);

    if (top.length === 0) {
      return res.status(200).json({ answer: "السؤال خارج نطاق الكيقن." });
    }

    const context = top.map((t, i) => `# مقطع ${i+1}: ${t.chunk.title}\n${t.chunk.content}`).join("\n\n");

    // 7) توليد الإجابة المقيدة بالسياق
    const chatRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST", headers,
      body: JSON.stringify({
        model: "gpt-4o-mini",
        temperature: 0.2,
        messages: [
          {
            role: "system",
            content:
`أنت مساعد ذكي مُقيّد بمجال "الكيقن".
أجب بالعربية بإيجاز ودقة، واستند فقط إلى المقاطع المرفقة.
إن كان السؤال خارج المجال أو لا يوجد سياق مناسب، أجب حرفيًا: "السؤال خارج نطاق الكيقن."`
          },
          { role: "user", content: `السؤال: ${question}\n\nالسياق:\n${context}\n\nأجب استناداً للسياق فقط.` }
        ]
      })
    });

    const data = await chatRes.json();
    const answer = data?.choices?.[0]?.message?.content?.trim() || "السؤال خارج نطاق الكيقن.";
    const sources = top.map(t => t.chunk.title);
    return res.status(200).json({ answer: `${answer}\n\nالمصادر: ${sources.join("، ")}` });

  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Server error" });
  }
}
