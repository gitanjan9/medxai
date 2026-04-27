import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, X, Send, Bot, User, Loader2 } from "lucide-react";
import type { PrimaryPrediction } from "../types";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
}

interface Props {
  context: PrimaryPrediction | null;
}

function formatMarkdown(text: string) {
  const parts = text.split(/\*\*(.*?)\*\*/g);
  return parts.map((p, i) =>
    i % 2 === 1 ? (
      <strong key={i} className="font-semibold text-slate-800">
        {p}
      </strong>
    ) : (
      <React.Fragment key={i}>
        {p.split("\n").map((line, j, arr) => (
          <React.Fragment key={j}>
            {line.startsWith("• ") ? (
              <span className="block pl-3 before:content-['•'] before:absolute before:left-0 relative">
                &nbsp;{line.slice(2)}
              </span>
            ) : (
              line
            )}
            {j < arr.length - 1 && line !== "" && <br />}
          </React.Fragment>
        ))}
      </React.Fragment>
    )
  );
}

const SUGGESTED = [
  "What did the model find?",
  "What does this mean clinically?",
  "What are the next steps?",
  "What biases affect this result?",
  "Why were other findings suppressed?",
  "How confident is the model?",
];

export default function ChatPanel({ context }: Props) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Reset on new analysis
  useEffect(() => {
    if (context) {
      setMessages([
        {
          id: "welcome",
          role: "assistant",
          content: `Analysis complete. Primary finding: **${context.label}** (${(context.calibrated_score * 100).toFixed(0)}% confidence, ${context.decision.replace(/_/g, " ")}).\n\nAsk me anything about this result — findings, evidence, next steps, or model biases.`,
        },
      ]);
    } else {
      setMessages([]);
    }
  }, [context?.label, context?.calibrated_score]);

  async function send(text: string) {
    if (!text.trim() || loading) return;
    setInput("");

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: text };
    const pendingMsg: Message = { id: "pending", role: "assistant", content: "", pending: true };

    setMessages((prev) => [...prev, userMsg, pendingMsg]);
    setLoading(true);

    const history = [...messages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const res = await fetch("/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history, context }),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = (data as { detail?: string }).detail ?? `Server error ${res.status}`;
        throw new Error(detail);
      }
      const reply: Message = {
        id: Date.now().toString() + "_r",
        role: "assistant",
        content: (data as { reply?: string }).reply ?? "No reply received.",
      };
      setMessages((prev) => [...prev.filter((m) => m.id !== "pending"), reply]);
    } catch (err) {
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== "pending"),
        {
          id: "err",
          role: "assistant",
          content: `Error: ${err instanceof Error ? err.message : "Connection failed — is the server running?"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  const hasContext = !!context;

  return (
    <>
      {/* Floating button */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-blue-600 hover:bg-blue-700 shadow-xl flex items-center justify-center transition-colors"
        aria-label="Open medical chat"
      >
        <AnimatePresence mode="wait">
          {open ? (
            <motion.span
              key="close"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: 90, opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <X className="w-6 h-6 text-white" />
            </motion.span>
          ) : (
            <motion.span
              key="open"
              initial={{ rotate: 90, opacity: 0 }}
              animate={{ rotate: 0, opacity: 1 }}
              exit={{ rotate: -90, opacity: 0 }}
              transition={{ duration: 0.15 }}
            >
              <MessageCircle className="w-6 h-6 text-white" />
            </motion.span>
          )}
        </AnimatePresence>
      </button>

      {/* Chat panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 24, scale: 0.97 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
            className="fixed bottom-24 right-6 z-50 w-[420px] max-h-[70vh] bg-white border border-slate-200 rounded-2xl shadow-2xl flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center gap-2.5 px-4 py-3 border-b border-slate-100 bg-slate-50">
              <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-slate-800">Medical Report Assistant</p>
                <p className="text-xs text-slate-400">
                  {hasContext
                    ? `Context: ${context!.label} analysis loaded`
                    : "Upload & analyse an X-ray to enable context-aware answers"}
                </p>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="text-slate-400 hover:text-slate-600 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 scrollbar-thin">
              {messages.length === 0 && (
                <div className="text-center py-6 space-y-2">
                  <Bot className="w-10 h-10 text-slate-200 mx-auto" />
                  <p className="text-xs text-slate-400">
                    {hasContext
                      ? "Ask a question about the current analysis."
                      : "Run an analysis first, then ask questions about the result."}
                  </p>
                </div>
              )}

              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex items-start gap-2 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                >
                  <div
                    className={`w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center ${
                      msg.role === "user" ? "bg-slate-200" : "bg-blue-600"
                    }`}
                  >
                    {msg.role === "user" ? (
                      <User className="w-3.5 h-3.5 text-slate-600" />
                    ) : (
                      <Bot className="w-3.5 h-3.5 text-white" />
                    )}
                  </div>
                  <div
                    className={`max-w-[80%] rounded-2xl px-3.5 py-2.5 text-xs leading-relaxed ${
                      msg.role === "user"
                        ? "bg-blue-600 text-white rounded-tr-sm"
                        : "bg-slate-50 text-slate-700 border border-slate-100 rounded-tl-sm"
                    }`}
                  >
                    {msg.pending ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-400" />
                    ) : msg.role === "user" ? (
                      msg.content
                    ) : (
                      formatMarkdown(msg.content)
                    )}
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>

            {/* Suggested questions */}
            {hasContext && messages.length <= 1 && !loading && (
              <div className="px-4 pb-2 flex flex-wrap gap-1.5">
                {SUGGESTED.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="text-xs bg-slate-50 hover:bg-blue-50 hover:text-blue-700 text-slate-500 border border-slate-200 rounded-full px-2.5 py-1 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            {/* Input */}
            <div className="px-4 py-3 border-t border-slate-100 flex items-end gap-2">
              <textarea
                ref={inputRef}
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKey}
                disabled={loading}
                placeholder={hasContext ? "Ask about this report…" : "Run an analysis first…"}
                className="flex-1 resize-none rounded-xl border border-slate-200 px-3 py-2 text-xs text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 leading-relaxed"
                style={{ maxHeight: 80 }}
              />
              <button
                onClick={() => send(input)}
                disabled={!input.trim() || loading}
                className="w-8 h-8 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-slate-200 disabled:cursor-not-allowed flex items-center justify-center transition-colors flex-shrink-0"
              >
                {loading ? (
                  <Loader2 className="w-3.5 h-3.5 text-white animate-spin" />
                ) : (
                  <Send className="w-3.5 h-3.5 text-white" />
                )}
              </button>
            </div>

            {/* Disclaimer */}
            <p className="text-[10px] text-slate-400 text-center px-4 pb-2.5 leading-tight">
              AI assistant only — not a substitute for professional clinical judgement.
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
