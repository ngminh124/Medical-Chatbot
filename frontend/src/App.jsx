import { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import { useAuth } from "./contexts/AuthContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import ProtectedRoute from "./components/ProtectedRoute";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import ChatPage from "./pages/ChatPage";
import AdminLayout from "./components/admin/AdminLayout";

const AdminOverviewPage = lazy(() => import("./pages/admin/OverviewPage"));
const AdminUsersPage = lazy(() => import("./pages/admin/UsersPage"));
const AdminConversationsPage = lazy(() => import("./pages/admin/ConversationsPage"));
const AdminMetricsPage = lazy(() => import("./pages/admin/MetricsPage"));
const AdminLogsPage = lazy(() => import("./pages/admin/LogsPage"));
const AdminTracesPage = lazy(() => import("./pages/admin/TracesPage"));
const AdminSettingsPage = lazy(() => import("./pages/admin/SettingsPage"));

function RoleLanding() {
  const { user } = useAuth();
  const role = (user?.type || "user").toLowerCase();
  return <Navigate to={role === "admin" ? "/admin" : "/chat"} replace />;
}

function PageLoader() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 text-slate-300">
      Loading...
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />

              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <RoleLanding />
                  </ProtectedRoute>
                }
              />

              <Route
                path="/chat"
                element={
                  <ProtectedRoute>
                    <ChatPage />
                  </ProtectedRoute>
                }
              />

              <Route
                path="/admin"
                element={
                  <ProtectedRoute allowedRoles={["admin"]} redirectTo="/chat">
                    <AdminLayout />
                  </ProtectedRoute>
                }
              >
                <Route index element={<AdminOverviewPage />} />
                <Route path="users" element={<AdminUsersPage />} />
                <Route path="conversations" element={<AdminConversationsPage />} />
                <Route path="metrics" element={<AdminMetricsPage />} />
                <Route path="logs" element={<AdminLogsPage />} />
                <Route path="traces" element={<AdminTracesPage />} />
                <Route path="settings" element={<AdminSettingsPage />} />
              </Route>

              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}
