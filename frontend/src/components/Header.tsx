import { useState } from "react";
import { ClipboardList, LayoutDashboard, LogOut, ChevronDown } from "lucide-react";
import { useAuth } from "../context/AuthContext";

type Page = "main" | "history";

interface HeaderProps {
  currentPage: Page;
  onNavigate: (page: Page) => void;
  historyCount?: number;
}

export default function Header({ currentPage, onNavigate, historyCount = 0 }: HeaderProps) {
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  async function handleLogout() {
    setLoggingOut(true);
    // Clear persisted prediction cache so the next user starts fresh
    sessionStorage.removeItem("medxai_last_prediction_id");
    sessionStorage.removeItem("medxai_last_image");
    try {
      await logout(); // calls POST /v1/auth/logout → clears session_id + refresh_token cookies
    } finally {
      setLoggingOut(false);
    }
  }

  const initials = user?.name
    ? user.name.split(" ").map((w) => w[0]).join("").slice(0, 2).toUpperCase()
    : user?.email?.[0]?.toUpperCase() ?? "?";

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
      <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center justify-between">
        {/* Left: logo + title */}
        <button
          onClick={() => onNavigate("main")}
          className="flex items-center gap-3 hover:opacity-80 transition-opacity"
        >
          <img
            src="/png-clipart-medical-symbol-who-logo-thumbnail.png"
            alt="Medical Symbol"
            className="w-9 h-9 flex-shrink-0"
          />
          <div className="text-left">
            <span className="text-sm font-semibold text-slate-900 tracking-tight">
              MedicalXAI
            </span>
            <span className="text-sm text-slate-400 font-normal ml-1.5">
              Clinician Workspace
            </span>
          </div>
        </button>

        {/* Right: nav + user menu */}
        <div className="flex items-center gap-1.5">
          <nav className="flex items-center gap-1.5">
            <button
              onClick={() => onNavigate("main")}
              className={`flex items-center gap-1.5 text-sm font-medium px-3.5 py-1.5 rounded-lg transition-colors ${
                currentPage === "main"
                  ? "bg-slate-100 text-slate-800"
                  : "text-slate-500 hover:bg-slate-50 hover:text-slate-700"
              }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              Workspace
            </button>

            <button
              onClick={() => onNavigate("history")}
              className={`flex items-center gap-1.5 text-sm font-medium px-3.5 py-1.5 rounded-lg transition-colors ${
                currentPage === "history"
                  ? "bg-slate-100 text-slate-800"
                  : "text-slate-500 hover:bg-slate-50 hover:text-slate-700"
              }`}
            >
              <ClipboardList className="w-4 h-4" />
              Patient Records
              {historyCount > 0 && (
                <span className="ml-1 text-xs font-semibold bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full">
                  {historyCount}
                </span>
              )}
            </button>
          </nav>

          {/* Divider */}
          <div className="w-px h-5 bg-slate-200 mx-1.5" />

          {/* User menu */}
          <div className="relative">
            <button
              onClick={() => setMenuOpen((v) => !v)}
              className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg hover:bg-slate-50 transition-colors"
            >
              <div className="w-7 h-7 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                {initials}
              </div>
              <div className="text-left hidden sm:block">
                <p className="text-xs font-semibold text-slate-700 leading-tight max-w-[120px] truncate">
                  {user?.name || user?.email}
                </p>
                <p className="text-xs text-slate-400 capitalize leading-tight">{user?.role}</p>
              </div>
              <ChevronDown className={`w-3.5 h-3.5 text-slate-400 transition-transform ${menuOpen ? "rotate-180" : ""}`} />
            </button>

            {/* Dropdown */}
            {menuOpen && (
              <>
                {/* Click-away backdrop */}
                <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
                <div className="absolute right-0 top-full mt-1.5 w-56 bg-white border border-slate-200 rounded-xl shadow-lg z-50 overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-100">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                        {initials}
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-slate-800 truncate">{user?.name}</p>
                        <p className="text-xs text-slate-400 truncate">{user?.email}</p>
                      </div>
                    </div>
                    <span className={`mt-2 inline-block text-xs font-medium px-2 py-0.5 rounded-full capitalize ${
                      user?.role === "admin"
                        ? "bg-purple-50 text-purple-700 border border-purple-200"
                        : "bg-blue-50 text-blue-700 border border-blue-200"
                    }`}>
                      {user?.role}
                    </span>
                  </div>

                  <div className="p-1.5">
                    <button
                      onClick={handleLogout}
                      disabled={loggingOut}
                      className="w-full flex items-center gap-2.5 px-3 py-2 text-sm text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                    >
                      <LogOut className="w-4 h-4" />
                      {loggingOut ? "Signing out…" : "Sign out"}
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
