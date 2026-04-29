import { Outlet } from "react-router-dom";
import AdminSidebar from "./AdminSidebar";
import AdminTopbar from "./AdminTopbar";

export default function AdminLayout() {
  return (
    <div className="flex min-h-screen bg-slate-950">
      <AdminSidebar />
      <div className="flex min-h-screen min-w-0 flex-1 flex-col">
        <AdminTopbar />
        <main className="min-w-0 flex-1 overflow-auto bg-slate-900/40 p-4 md:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
