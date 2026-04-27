import { useState } from "react";
import { motion } from "framer-motion";
import {
  Eye, EyeOff, Loader2, UserPlus, AlertCircle, CheckCircle2, Check, X,
} from "lucide-react";
import { authApi, setAccessToken } from "../services/api";
import { useAuth } from "../context/AuthContext";

interface Props {
  onLoginClick: () => void;
}

interface PasswordRule {
  label: string;
  test: (pw: string) => boolean;
}

const PASSWORD_RULES: PasswordRule[] = [
  { label: "At least 8 characters",        test: (p) => p.length >= 8 },
  { label: "At least one uppercase letter", test: (p) => /[A-Z]/.test(p) },
  { label: "At least one number",           test: (p) => /[0-9]/.test(p) },
  { label: "At least one special character",test: (p) => /[^A-Za-z0-9]/.test(p) },
];

export default function RegisterPage({ onLoginClick }: Props) {
  const { login } = useAuth();

  const [name, setName]           = useState("");
  const [email, setEmail]         = useState("");
  const [password, setPassword]   = useState("");
  const [confirm, setConfirm]     = useState("");
  const [showPw, setShowPw]       = useState(false);
  const [showCf, setShowCf]       = useState(false);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [success, setSuccess]     = useState(false);

  const pwRulesMet  = PASSWORD_RULES.filter((r) => r.test(password));
  const pwStrength  = pwRulesMet.length;           // 0–4
  const allPwRules  = pwStrength === PASSWORD_RULES.length;
  const confirmsMatch = password === confirm && confirm.length > 0;

  const strengthColor = ["bg-slate-200", "bg-red-400", "bg-orange-400", "bg-yellow-400", "bg-green-500"][pwStrength];
  const strengthLabel = ["", "Weak", "Fair", "Good", "Strong"][pwStrength];

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || !email || !password || !confirm) {
      setError("All fields are required.");
      return;
    }
    if (!allPwRules) {
      setError("Password does not meet all requirements.");
      return;
    }
    if (!confirmsMatch) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Register account
      await authApi.register(email.trim(), password, name.trim());
      setSuccess(true);

      // Auto-login after registration
      await login(email.trim(), password);
    } catch (err) {
      setSuccess(false);
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  }

  if (success) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white border border-slate-200 rounded-2xl shadow-lg p-10 flex flex-col items-center gap-4 max-w-sm w-full"
        >
          <div className="w-14 h-14 rounded-full bg-green-100 flex items-center justify-center">
            <CheckCircle2 className="w-7 h-7 text-green-600" />
          </div>
          <h2 className="text-lg font-semibold text-slate-800">Account created!</h2>
          <p className="text-sm text-slate-500 text-center">
            Welcome, <span className="font-medium text-slate-700">{name}</span>.<br />
            Signing you in…
          </p>
          <div className="w-5 h-5 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin mt-2" />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: "easeOut" }}
        className="w-full max-w-md"
      >
        {/* Logo */}
        <div className="flex flex-col items-center mb-8 gap-3">
          <img
            src="/png-clipart-medical-symbol-who-logo-thumbnail.png"
            alt="MedicalXAI"
            className="w-14 h-14"
          />
          <div className="text-center">
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">MedicalXAI</h1>
            <p className="text-sm text-slate-400 mt-0.5">Create your clinician account</p>
          </div>
        </div>

        {/* Card */}
        <div className="bg-white border border-slate-200 rounded-2xl shadow-lg p-8">
          <div className="flex items-center gap-2 mb-6">
            <UserPlus className="w-5 h-5 text-blue-500" />
            <h2 className="text-base font-semibold text-slate-800">New account</h2>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-start gap-2.5 bg-red-50 border border-red-200 rounded-xl px-4 py-3 mb-5"
            >
              <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-700">{error}</p>
            </motion.div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            {/* Full name */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Full name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Dr. Jane Smith"
                autoComplete="name"
                required
                className="w-full text-sm px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white placeholder:text-slate-300 transition-shadow"
              />
            </div>

            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Email address</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="clinician@hospital.org"
                autoComplete="email"
                required
                className="w-full text-sm px-3.5 py-2.5 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white placeholder:text-slate-300 transition-shadow"
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Password</label>
              <div className="relative">
                <input
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  autoComplete="new-password"
                  required
                  className="w-full text-sm px-3.5 py-2.5 pr-10 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white placeholder:text-slate-300 transition-shadow"
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                  tabIndex={-1}
                >
                  {showPw ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>

              {/* Strength bar */}
              {password.length > 0 && (
                <div className="mt-2">
                  <div className="flex gap-1 mb-1.5">
                    {[0, 1, 2, 3].map((i) => (
                      <div
                        key={i}
                        className={`h-1 flex-1 rounded-full transition-colors ${i < pwStrength ? strengthColor : "bg-slate-100"}`}
                      />
                    ))}
                  </div>
                  <p className={`text-xs font-medium ${["", "text-red-500", "text-orange-500", "text-yellow-600", "text-green-600"][pwStrength]}`}>
                    {strengthLabel}
                  </p>
                </div>
              )}

              {/* Rules checklist */}
              {password.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {PASSWORD_RULES.map((rule) => {
                    const met = rule.test(password);
                    return (
                      <li key={rule.label} className={`flex items-center gap-1.5 text-xs ${met ? "text-green-600" : "text-slate-400"}`}>
                        {met
                          ? <Check className="w-3 h-3 flex-shrink-0" />
                          : <X className="w-3 h-3 flex-shrink-0" />}
                        {rule.label}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            {/* Confirm password */}
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">Confirm password</label>
              <div className="relative">
                <input
                  type={showCf ? "text" : "password"}
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder="••••••••"
                  autoComplete="new-password"
                  required
                  className={`w-full text-sm px-3.5 py-2.5 pr-10 border rounded-xl focus:outline-none focus:ring-2 focus:border-transparent bg-white placeholder:text-slate-300 transition-shadow ${
                    confirm.length > 0
                      ? confirmsMatch
                        ? "border-green-300 focus:ring-green-400"
                        : "border-red-300 focus:ring-red-400"
                      : "border-slate-200 focus:ring-blue-500"
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowCf((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                  tabIndex={-1}
                >
                  {showCf ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {confirm.length > 0 && !confirmsMatch && (
                <p className="mt-1 text-xs text-red-500">Passwords do not match</p>
              )}
              {confirmsMatch && (
                <p className="mt-1 text-xs text-green-600 flex items-center gap-1">
                  <Check className="w-3 h-3" /> Passwords match
                </p>
              )}
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold text-sm py-2.5 rounded-xl transition-colors mt-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Creating account…
                </>
              ) : (
                <>
                  <UserPlus className="w-4 h-4" />
                  Create account
                </>
              )}
            </button>
          </form>
        </div>

        {/* Footer: link to login */}
        <p className="text-center text-sm text-slate-500 mt-5">
          Already have an account?{" "}
          <button
            onClick={onLoginClick}
            className="text-blue-600 hover:text-blue-700 font-medium underline-offset-2 hover:underline"
          >
            Sign in
          </button>
        </p>
      </motion.div>
    </div>
  );
}
