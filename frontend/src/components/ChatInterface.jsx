// src/components/ChatInterface.jsx
import React, { useState, useRef, useEffect } from "react";
import {
  Bell,
  FileText,
  Settings,
  Info,
  User,
  LogOut,
  LayoutDashboard,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import sendIcon from "../../icons/send.svg";
import voiceIcon from "../../icons/voice_mode.svg";
import dexterIcon from "../../icons/dexter.svg";
import micIcon from "../../icons/microphone.svg";
import newChatIcon from "../../icons/newchat.svg";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const ChatInterface = () => {
  const [currentView, setCurrentView] = useState("welcome"); // 'welcome' | 'chatting'
  const [messages, setMessages] = useState([]); // chat history
  const [thinking, setThinking] = useState([]); // temporary "trace"
  const [inputValue, setInputValue] = useState("");
  const [recording, setRecording] = useState(false);
  const [showProfileDropdown, setShowProfileDropdown] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);
  const chatContainerRef = useRef(null);
  const navigate = useNavigate();
  const inputBarRef = useRef(null);

  // Auto-scroll to bottom with proper padding calculation
  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      const container = chatContainerRef.current;
      const inputBarHeight = inputBarRef.current?.offsetHeight || 96;
      container.scrollTo({
        top: container.scrollHeight - container.clientHeight + inputBarHeight,
        behavior: "smooth",
      });
    }
  };

  useEffect(() => {
    if (currentView === "chatting") {
      const timer = setTimeout(scrollToBottom, 50);
      return () => clearTimeout(timer);
    }
  }, [messages, thinking, currentView]);

  // Logout
  const handleLogout = () => {
    localStorage.removeItem("authToken");
    localStorage.removeItem("userRole");
    localStorage.removeItem("userRegNo");
    navigate("/");
  };

  const handleGoToDashboard = () => {
    navigate("/dashboard");
  };

  const isAdmin = localStorage.getItem("userRole") === "admin";

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest(".profile-dropdown")) {
        setShowProfileDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    const handleResize = () => {
      if (currentView === "chatting") {
        setTimeout(scrollToBottom, 100);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [currentView]);

  // ---------------- TEXT CHAT STREAM ----------------
  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = inputValue;
    setMessages((prev) => [...prev, { type: "user", text: userMessage }]);
    setInputValue("");
    setCurrentView("chatting");

    setTimeout(scrollToBottom, 100);

    const botIndex = messages.length + 1;
    setMessages((prev) => [...prev, { type: "bot", text: "" }]);
    setThinking([]);

    try {
      const department = localStorage.getItem("userDept") || "General";
      const userId = localStorage.getItem("userRegNo") || "default";
      const token = localStorage.getItem("authToken");

      const res = await fetch("http://127.0.0.1:8000/chat_stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          message: userMessage,
          department,
          k: 5,
          user_id: userId,
        }),
      });

      if (!res.ok || !res.body) {
        throw new Error("Backend not reachable");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let botText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const lines = decoder.decode(value, { stream: true }).split("\n");
        for (let line of lines) {
          if (!line.trim()) continue;
          const evt = JSON.parse(line);

          if (evt.type === "status") {
            setThinking((prev) => [...prev, { kind: "status", text: evt.message }]);
          } else if (evt.type === "doc") {
            setThinking((prev) => [
              ...prev,
              { kind: "doc", text: `🔗 ${evt.source}: ${evt.preview}` },
            ]);
          } else if (evt.type === "token") {
            botText += evt.text;
            setMessages((prev) =>
              prev.map((m, i) => (i === botIndex ? { ...m, text: botText } : m))
            );
            scrollToBottom();
          } else if (evt.type === "done") {
            setThinking([]);
          }
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      setMessages((prev) =>
        prev.map((m, i) =>
          i === botIndex ? { ...m, text: "⚠️ Error connecting to backend." } : m
        )
      );
      setThinking([]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSendMessage();
  };

  // ---------------- VOICE CHAT ----------------
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);

      audioChunks.current = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunks.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("file", audioBlob, "query.wav");

        setMessages((prev) => [
          ...prev,
          { type: "user", text: "🎤 (voice query sent)" },
        ]);
        setCurrentView("chatting");
        setTimeout(scrollToBottom, 100);

        try {
          const token = localStorage.getItem("authToken");
          const res = await fetch("http://127.0.0.1:8000/voice_chat", {
            method: "POST",
            headers: {
              ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
            body: formData,
          });

          if (!res.ok) throw new Error("Voice chat failed");
          const audioBlobResp = await res.blob();

          const audioURL = URL.createObjectURL(audioBlobResp);
          const audio = new Audio(audioURL);
          audio.play();

          setMessages((prev) => [
            ...prev,
            { type: "bot", text: "🔊 Played response audio" },
          ]);
          setTimeout(scrollToBottom, 100);
        } catch (err) {
          console.error("Voice chat error:", err);
          setMessages((prev) => [
            ...prev,
            { type: "bot", text: "❌ Voice chat error." },
          ]);
          setTimeout(scrollToBottom, 100);
        }
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setRecording(true);
    } catch (err) {
      alert("Microphone access denied or unsupported.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const toggleRecording = () => {
    if (recording) stopRecording();
    else startRecording();
  };

  // ---------------- RENDER ----------------
  return (
    <div className="h-screen w-full flex flex-col bg-gray-100 relative">
      {/* Header */}
      <div className="flex justify-between items-center px-8 py-4 bg-gray-100 border-b border-gray-200 z-20 relative">
        <span className="px-3 py-1 bg-white rounded-md text-sm font-normal shadow">
          Computer Science Department
        </span>
        <span className="text-sm text-gray-400 mx-auto">
          Emergency contacts | Phone: +91 1800 123 0001 | Email:
          helpdesk@polyuni.edu
        </span>

        {/* Profile Dropdown */}
        <div className="relative profile-dropdown">
          <button
            onClick={() => setShowProfileDropdown(!showProfileDropdown)}
            className="w-10 h-10 flex items-center justify-center rounded-full bg-gray-800 hover:bg-gray-700 transition-colors"
          >
            <User size={20} className="text-white" />
          </button>

          {showProfileDropdown && (
            <div className="absolute right-0 top-12 bg-white rounded-lg shadow-lg border border-gray-200 py-2 min-w-[160px] z-30">
              <div className="px-4 py-2 text-sm text-gray-600 border-b border-gray-100">
                {localStorage.getItem("userRegNo") || "User"}
              </div>
              {isAdmin && (
                <button
                  onClick={handleGoToDashboard}
                  className="w-full px-4 py-2 text-left text-sm text-blue-600 hover:bg-blue-50 flex items-center gap-2 transition-colors"
                >
                  <LayoutDashboard size={16} />
                  Go to Dashboard
                </button>
              )}
              <button
                onClick={handleLogout}
                className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors"
              >
                <LogOut size={16} />
                Sign out
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Dexter floating avatar */}
      <div
        className="absolute top-1/3 right-0 flex items-center bg-white rounded-l-full shadow-md px-4 py-2 gap-2 z-10"
        style={{ transform: "translateY(-50%)" }}
      >
        <img src={dexterIcon} alt="Dexter" className="w-12 h-12 rounded-full" />
        <span className="text-sm font-medium">Dexter</span>
      </div>

      {/* New Chat floating button (opposite Dexter) */}
      <div
        className="absolute top-1/3 left-0 flex items-center bg-white rounded-r-full shadow-md px-4 py-2 gap-2 z-10 cursor-pointer hover:bg-gray-50"
        style={{ transform: "translateY(-50%)" }}
        onClick={() => {
          setMessages([]);
          setThinking([]);
          setInputValue("");
          setCurrentView("welcome");
        }}
      >
        <img src={newChatIcon} alt="New Chat" className="w-6 h-6" />
        <span className="text-sm font-medium">New Chat</span>
      </div>



      {/* Content Container */}
      <div className="flex-1 w-full max-w-4xl mx-auto flex flex-col relative min-h-0">
        {currentView === "welcome" ? (
          <div className="flex-1 flex flex-col items-center justify-center p-6">
            <h1 className="text-3xl font-semibold mb-8">
              How can I help you today?
            </h1>
            <div className="flex gap-6 mb-12">
              <div className="bg-white border rounded-xl p-6 w-56 text-center shadow-sm hover:shadow-md cursor-pointer transition-shadow">
                <Bell className="w-8 h-8 mx-auto mb-3 text-gray-600" />
                <h3 className="font-medium text-gray-900">Learn about</h3>
                <p className="text-sm text-gray-600">updated policies</p>
              </div>
              <div className="bg-white border rounded-xl p-6 w-56 text-center shadow-sm hover:shadow-md cursor-pointer transition-shadow">
                <FileText className="w-8 h-8 mx-auto mb-3 text-gray-600" />
                <h3 className="font-medium text-gray-900">Review the fee</h3>
                <p className="text-sm text-gray-600">structures in brief</p>
              </div>
              <div className="bg-white border rounded-xl p-6 w-56 text-center shadow-sm hover:shadow-md cursor-pointer transition-shadow">
                <Settings className="w-8 h-8 mx-auto mb-3 text-gray-600" />
                <h3 className="font-medium text-gray-900">Verify your</h3>
                <p className="text-sm text-gray-600">eligibility criteria</p>
              </div>
            </div>
          </div>
        ) : (
          <div
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto bg-gray-100 min-h-0"
            style={{
              backgroundColor: "#f3f4f6",
              paddingBottom: "120px",
              paddingTop: "1rem",
              paddingLeft: "1.5rem",
              paddingRight: "1.5rem",
              scrollbarWidth: "thin",
              scrollbarColor: "rgba(156, 163, 175, 0.3) transparent",
            }}
          >
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${
                  msg.type === "user" ? "justify-end" : "justify-start"
                } mb-4`}
              >
                <div
                  className={`px-4 py-3 rounded-lg max-w-lg whitespace-pre-wrap ${
                    msg.type === "user"
                      ? "bg-gray-900 text-white"
                      : "bg-white text-black shadow-sm border border-gray-200"
                  }`}
                >
                  {msg.type === "bot" ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        h1: ({ node, ...props }) => (
                          <h1 className="text-2xl font-bold mb-3" {...props} />
                        ),
                        h2: ({ node, ...props }) => (
                          <h2
                            className="text-xl font-semibold mt-4 mb-2"
                            {...props}
                          />
                        ),
                        h3: ({ node, ...props }) => (
                          <h3
                            className="text-lg font-medium mt-3 mb-1"
                            {...props}
                          />
                        ),
                        ul: ({ node, ...props }) => (
                          <ul
                            className="list-disc list-inside space-y-1 my-2"
                            {...props}
                          />
                        ),
                        ol: ({ node, ...props }) => (
                          <ol
                            className="list-decimal list-inside space-y-1 my-2"
                            {...props}
                          />
                        ),
                        li: ({ node, ...props }) => (
                          <li className="ml-2" {...props} />
                        ),
                        strong: ({ node, ...props }) => (
                          <strong
                            className="font-semibold text-black"
                            {...props}
                          />
                        ),
                        p: ({ node, ...props }) => (
                          <p className="mb-2" {...props} />
                        ),
                        hr: () => <hr className="my-4 border-gray-300" />,
                        code: ({ node, inline, ...props }) =>
                          inline ? (
                            <code
                              className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono"
                              {...props}
                            />
                          ) : (
                            <code
                              className="block bg-gray-100 p-3 rounded text-sm font-mono overflow-x-auto"
                              {...props}
                            />
                          ),
                      }}
                    >
                      {msg.text}
                    </ReactMarkdown>
                  ) : (
                    msg.text
                  )}
                </div>
              </div>
            ))}

            {thinking.length > 0 && (
              <div className="flex justify-start mb-4">
                <div className="px-4 py-3 rounded-lg max-w-lg bg-yellow-50 text-gray-700 text-sm border border-yellow-200 shadow-sm">
                  {thinking.map((t, idx) => (
                    <div key={idx} className="mb-1 last:mb-0">
                      {t.text}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Input bar */}
        <div
          ref={inputBarRef}
          className="fixed bottom-0 left-0 right-0 bg-gray-100 border-t border-gray-200 z-30"
          style={{ backgroundColor: "#f3f4f6" }}
        >
          <div className="w-full max-w-4xl mx-auto flex items-center p-4 gap-4">
            <div className="flex-1 relative">
              <input
                type="text"
                placeholder="Ask me anything..."
                className="w-full px-6 py-4 bg-black text-white rounded-full pr-24 focus:outline-none focus:ring-2 focus:ring-gray-600 text-base shadow-lg"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
              />
              <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-1 items-center">
                <button
                  className="w-10 h-10 rounded-full flex items-center justify-center hover:bg-gray-800 transition-colors"
                  onClick={() => {
                    /* Future voice mode button */
                  }}
                >
                  <img src={voiceIcon} alt="Voice" className="w-6 h-6" />
                </button>
                <button
                  className="w-10 h-10 rounded-full flex items-center justify-center hover:bg-gray-800 transition-colors"
                  onClick={
                    inputValue.trim() ? handleSendMessage : toggleRecording
                  }
                >
                  {inputValue.trim() ? (
                    <img src={sendIcon} alt="Send" className="w-6 h-6" />
                  ) : (
                    <img
                      src={micIcon}
                      alt="Microphone"
                      className={`w-6 h-6 ${
                        recording ? "animate-pulse opacity-75" : ""
                      }`}
                    />
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
