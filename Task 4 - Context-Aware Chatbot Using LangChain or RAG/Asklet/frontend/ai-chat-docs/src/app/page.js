"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { BsSendFill } from "react-icons/bs";
import { FiUploadCloud } from "react-icons/fi";

export default function Home() {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);
  const [pdfId, setPdfId] = useState("");
  const [uploading, setUploading] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  const BASE_URL = "http://localhost:8010/api";

  const getTimestamp = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  useEffect(() => {
    scrollToBottom();
  }, [chat]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${BASE_URL}/upload_pdf/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setPdfId(res.data.pdf_id);
      setChat([
        {
          type: "bot",
          text: `📄 PDF *${file.name}* is all set! Asklet is on standby — go ahead and ask any question based on this document.`,
          timestamp: getTimestamp(),
        },
      ]);
    } catch (err) {
      alert("Failed to upload PDF.");
    } finally {
      setUploading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || !pdfId) return;

    const userMessage = {
      type: "user",
      text: input,
      timestamp: getTimestamp(),
    };

    setChat((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${BASE_URL}/chat/`, null, {
        params: { pdf_id: pdfId, query: input },
      });

      // const botMessage = {
      //   type: "bot",
      //   text: res.data.answer || "No answer received.",
      //   timestamp: getTimestamp(),
      // };

      // setChat((prev) => [...prev, botMessage]);
      // Extract answer
      const botMessage = {
        type: "bot",
        text: res.data.answer || "No answer received.",
        timestamp: getTimestamp(),
      };

      // Append both user and bot messages to local chat UI
      setChat((prev) => [...prev, botMessage]);

      // Optional: log history if you want debugging
      console.log("Conversation history:", res.data.chat_history);

    } catch (err) {
      setChat((prev) => [
        ...prev,
        {
          type: "bot",
          text: "❌ Failed to fetch response.",
          timestamp: getTimestamp(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-gray-100 to-blue-100 font-sans">
      <header className="bg-blue-700 text-white text-center py-4 text-2xl font-bold shadow-md sticky top-0 z-20">
        📚 Asklet
      </header>

      {/* Upload Area */}
      <section className="px-4 py-6 bg-white shadow-md border-b">
        <div className="max-w-3xl mx-auto text-center">
          <label
            htmlFor="pdfUpload"
            className="cursor-pointer inline-flex items-center gap-3 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl font-semibold shadow-md transition"
          >
            <FiUploadCloud className="text-xl" />
            {uploading ? "Uploading..." : "Upload PDF"}
          </label>
          <input
            id="pdfUpload"
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={handleUpload}
          />
          <p className="mt-2 text-sm text-gray-500">
            {pdfId ? `Document ID: ${pdfId}` : "Please upload a PDF to start chatting."}
          </p>
        </div>
      </section>

      {/* Chat Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {chat.map((msg, i) => (
            <div
              key={i}
              className={`flex gap-2 ${msg.type === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.type === "bot" && (
                <div className="w-9 h-9 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm font-bold shadow">
                  🤖
                </div>
              )}

              <div
                className={`relative px-5 py-3 rounded-2xl text-base leading-relaxed shadow-md max-w-[80%] mb-4 ${msg.type === "user"
                  ? "bg-gradient-to-br from-blue-500 to-blue-700 text-white rounded-br-none"
                  : "bg-white text-gray-900 rounded-bl-none border border-gray-200"
                  }`}
                style={{ whiteSpace: "pre-wrap" }}
              >
                {msg.type === "bot" ? (
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                ) : (
                  <p className="font-medium">{msg.text}</p>
                )}
                <div className="absolute bottom-[-1.2rem] right-2 text-xs text-gray-500">
                  {msg.timestamp}
                </div>
              </div>

              {msg.type === "user" && (
                <div className="w-9 h-9 rounded-full bg-gray-800 text-white flex items-center justify-center text-sm font-bold shadow">
                  🧑
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex justify-start gap-2">
              <div className="w-9 h-9 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm font-bold shadow">
                🤖
              </div>
              <div className="px-4 py-2 bg-white rounded-xl shadow border border-gray-300 text-gray-700 text-sm font-medium">
                <span className="animate-pulse">Typing...</span>
              </div>
            </div>
          )}

          <div ref={chatEndRef} className="pt-6" />
        </div>
      </main>

      {/* Chat Input */}
      <footer className="bg-white border-t p-4 shadow-inner sticky bottom-0 z-10">
        <form onSubmit={handleSubmit} className="flex max-w-3xl mx-auto gap-3">
          <input
            ref={inputRef}
            type="text"
            disabled={!pdfId}
            placeholder={
              pdfId ? "Ask something about your PDF..." : "Upload a PDF first..."
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 p-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            type="submit"
            disabled={loading || !pdfId}
            className="ml-2 p-3 rounded-full bg-blue-600 hover:bg-blue-700 text-white shadow disabled:opacity-50"
          >
            <BsSendFill className="text-xl" />
          </button>
        </form>
      </footer>
    </div>
  );
}
