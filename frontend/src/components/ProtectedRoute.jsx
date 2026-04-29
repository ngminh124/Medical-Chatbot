import { Navigate } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { Loader2 } from "lucide-react";

export default function ProtectedRoute({ children, allowedRoles, redirectTo = "/login" }) {
  const { isAuthenticated, loading, user } = useAuth();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <Loader2 className="h-8 w-8 animate-spin text-primary-600" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (Array.isArray(allowedRoles) && allowedRoles.length > 0) {
    const role = (user?.type || "user").toLowerCase();
    const isAllowed = allowedRoles.map((r) => r.toLowerCase()).includes(role);
    if (!isAllowed) {
      return <Navigate to={redirectTo} replace />;
    }
  }

  return children;
}
