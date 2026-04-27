import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { authApi, setAccessToken, type ApiUser } from "../services/api";

interface AuthContextValue {
  user: ApiUser | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<ApiUser | null>(null);
  const [loading, setLoading] = useState(true);

  // On page load: use the HttpOnly refresh cookie to silently restore the
  // access token into memory. No cookie reading from JS — the browser sends
  // the cookie automatically in the POST /v1/auth/refresh request.
  useEffect(() => {
    authApi
      .refresh()
      .then((res) => {
        setAccessToken(res.access_token);   // stored in JS memory only
        setUser(res.user);
      })
      .catch(() => {
        setAccessToken(null);
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const res = await authApi.login(email, password);
    setAccessToken(res.access_token);       // into memory, not localStorage
    setUser(res.user);
  }, []);

  const logout = useCallback(async () => {
    await authApi.logout().catch(() => null);
    setAccessToken(null);                   // wipe from memory
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
