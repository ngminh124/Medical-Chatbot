import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  HeartPulse,
  Hospital,
  Pill,
  Sparkles,
} from "lucide-react";
import ThemeToggleButton from "../components/ThemeToggleButton";

const FEATURES = [
  {
    title: "Hỏi đáp 24/7",
    description: "Phản hồi nhanh, ổn định và dễ hiểu.",
    icon: HeartPulse,
  },
  {
    title: "Tra cứu triệu chứng",
    description: "Gợi ý thông minh theo biểu hiện.",
    icon: Activity,
  },
  {
    title: "Tư vấn thuốc",
    description: "Thông tin cơ sở, rõ ràng và cẩn thận.",
    icon: Pill,
  },
];

const STEPS = [
  {
    title: "Đăng ký",
    description: "Tạo tài khoản trong vài giây.",
    icon: Sparkles,
  },
  {
    title: "Đặt câu hỏi",
    description: "Mô tả triệu chứng hoặc nhu cầu.",
    icon: Hospital,
  },
  {
    title: "Nhận tư vấn",
    description: "Nhận thông tin y khoa để tham khảo.",
    icon: HeartPulse,
  },
];


export default function LandingPage() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-sky-50 via-white to-emerald-50 text-slate-900 dark:from-slate-950 dark:via-slate-950 dark:to-slate-900 dark:text-slate-100">
      <ThemeToggleButton />

      <div className="pointer-events-none absolute -top-32 right-0 h-72 w-72 rounded-full bg-primary-200/40 blur-3xl dark:bg-primary-900/40 animate-float-slow" />
      <div className="pointer-events-none absolute -bottom-40 left-0 h-96 w-96 rounded-full bg-emerald-200/30 blur-3xl dark:bg-emerald-900/30 animate-float-slower" />

      <header className="relative z-10 mx-auto flex w-full max-w-6xl items-center justify-between px-6 pb-6 pt-8">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary-600 text-white shadow-lg shadow-primary-200/60 dark:shadow-black/40">
            <HeartPulse className="h-6 w-6" />
          </div>
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-primary-600 dark:text-primary-300">
              Minqes
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">Medical RAG Chatbot</p>
          </div>
        </div>
        <div className="hidden items-center gap-3 sm:flex">
          <Link
            to="/login"
            className="rounded-full border border-slate-200 px-5 py-2 text-sm font-semibold text-slate-700 transition hover:border-primary-300 hover:text-primary-600 dark:border-slate-700 dark:text-slate-200 dark:hover:border-primary-400 dark:hover:text-primary-300"
          >
            Đăng nhập
          </Link>
          <Link
            to="/register"
            className="rounded-full bg-primary-600 px-5 py-2 text-sm font-semibold text-white shadow-lg shadow-primary-200/60 transition hover:bg-primary-700 dark:shadow-black/40"
          >
            Đăng ký
          </Link>
        </div>
      </header>

      <main className="relative z-10">
        <section className="mx-auto flex min-h-screen w-full max-w-6xl flex-col justify-center gap-12 px-6 pb-14 pt-8">
          <div className="grid items-center gap-12 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 rounded-full border border-primary-200 bg-white/80 px-4 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-primary-600 shadow-sm dark:border-primary-700 dark:bg-slate-900/70 dark:text-primary-200">
                <Sparkles className="h-4 w-4" />
                Tư vấn y khoa thông minh
              </div>
              <h1 className="font-display text-5xl font-semibold leading-tight text-slate-900 dark:text-white sm:text-6xl lg:text-7xl">
                Trợ lý y tế đáng tin cậy cho mọi câu hỏi hàng ngày.
              </h1>
              <p className="max-w-xl text-lg text-slate-600 dark:text-slate-300 sm:text-xl">
                Minqes tổng hợp kiến thức y khoa để hỗ trợ bạn nhanh, rõ ràng và dễ áp dụng.
              </p>
              <div className="flex flex-wrap items-center gap-4">
                <Link
                  to="/register"
                  className="group inline-flex items-center gap-3 rounded-full bg-primary-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-primary-200/70 transition hover:bg-primary-700 dark:shadow-black/40"
                >
                  Bắt đầu ngay
                  <ArrowRight className="h-4 w-4 transition group-hover:translate-x-1" />
                </Link>
                <Link
                  to="/login"
                  className="inline-flex items-center gap-2 rounded-full border border-slate-200 px-6 py-3 text-sm font-semibold text-slate-700 transition hover:border-primary-300 hover:text-primary-600 dark:border-slate-700 dark:text-slate-200 dark:hover:border-primary-400 dark:hover:text-primary-300"
                >
                  Đã có tài khoản
                </Link>
              </div>
            </div>
            <div className="relative">
              <div className="absolute -left-8 top-8 h-56 w-56 rounded-3xl border border-white/60 bg-white/70 shadow-2xl shadow-primary-200/50 backdrop-blur dark:border-white/10 dark:bg-slate-900/60 dark:shadow-black/40 animate-float-slow" />
              <div className="absolute -bottom-12 right-6 h-64 w-64 rounded-[32px] border border-white/60 bg-gradient-to-br from-primary-100/70 to-emerald-100/70 shadow-2xl shadow-emerald-200/60 backdrop-blur dark:border-white/10 dark:from-primary-900/40 dark:to-emerald-900/40 dark:shadow-black/40 animate-float-slower" />
              <div className="relative rounded-[36px] border border-white/70 bg-white/80 p-10 shadow-2xl shadow-primary-200/70 backdrop-blur dark:border-white/10 dark:bg-slate-900/70 dark:shadow-black/50">
                <div className="flex items-center gap-4">
                  <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-600 text-white">
                    <HeartPulse className="h-7 w-7" />
                  </div>
                  <div>
                    <p className="text-base font-semibold text-slate-900 dark:text-white">Minqes Assistant</p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">Kết nối tư vấn y khoa</p>
                  </div>
                </div>
                <div className="mt-6 space-y-4">
                  {[
                    "Tôi bị ho lâu ngày, nên làm gì?",
                    "Triệu chứng sốt và đau đầu có nguy hiểm không?",
                    "Cách dùng thuốc hạ sốt an toàn?",
                  ].map((text) => (
                    <div
                      key={text}
                      className="rounded-2xl border border-slate-200 bg-white/90 px-5 py-3 text-sm text-slate-700 shadow-sm dark:border-slate-700 dark:bg-slate-900/80 dark:text-slate-200"
                    >
                      {text}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="grid gap-8 lg:grid-cols-2">
            <div className="rounded-[28px] border border-slate-200 bg-white/85 p-8 shadow-lg shadow-slate-200/60 dark:border-slate-800 dark:bg-slate-900/70 dark:shadow-black/40">
              <div className="mb-5 flex items-center justify-between">
                <h2 className="font-display text-2xl font-semibold text-slate-900 dark:text-white sm:text-3xl">
                  Tính năng nổi bật
                </h2>
                <span className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  Cập nhật
                </span>
              </div>
              <div className="grid gap-5 sm:grid-cols-3">
                {FEATURES.map((feature) => {
                  const Icon = feature.icon;
                  return (
                    <div
                      key={feature.title}
                      className="rounded-2xl border border-slate-200 bg-white/90 p-5 shadow-sm transition hover:-translate-y-1 dark:border-slate-800 dark:bg-slate-900/70"
                    >
                      <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-2xl bg-primary-100 text-primary-600 dark:bg-primary-900/40 dark:text-primary-300">
                        <Icon className="h-5 w-5" />
                      </div>
                      <h3 className="text-base font-semibold text-slate-900 dark:text-white">
                        {feature.title}
                      </h3>
                      <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
                        {feature.description}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="rounded-[28px] border border-primary-100 bg-gradient-to-r from-white via-sky-50 to-emerald-50 p-8 shadow-lg shadow-primary-200/60 dark:border-slate-800 dark:from-slate-900/70 dark:via-slate-900/80 dark:to-slate-950 dark:shadow-black/40">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="font-display text-2xl font-semibold text-slate-900 dark:text-white sm:text-3xl">
                    Cách Minqes hoạt động
                  </h2>
                  <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                    Ba bước để bắt đầu
                  </p>
                </div>
                <Link
                  to="/register"
                  className="inline-flex items-center justify-center rounded-full bg-primary-600 px-5 py-2.5 text-xs font-semibold text-white shadow-lg shadow-primary-200/70 transition hover:bg-primary-700 dark:shadow-black/40"
                >
                  Đăng ký ngay
                </Link>
              </div>
              <div className="grid gap-5 sm:grid-cols-3">
                {STEPS.map((step, index) => {
                  const Icon = step.icon;
                  return (
                    <div
                      key={step.title}
                      className="rounded-2xl border border-white/80 bg-white/90 p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900/70"
                    >
                      <div className="mb-3 flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary-100 text-primary-600 dark:bg-primary-900/40 dark:text-primary-300">
                          <Icon className="h-4 w-4" />
                        </div>
                        <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                          Bước {index + 1}
                        </span>
                      </div>
                      <h3 className="text-base font-semibold text-slate-900 dark:text-white">
                        {step.title}
                      </h3>
                      <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
                        {step.description}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="relative z-10 border-t border-slate-200/80 bg-white/80 px-6 py-8 text-sm text-slate-500 backdrop-blur dark:border-slate-800 dark:bg-slate-950/80 dark:text-slate-400">
        <div className="mx-auto flex w-full max-w-6xl flex-col items-center justify-between gap-4 sm:flex-row">
          <p>
            Minqes - Medical RAG Chatbot
          </p>
          <div className="flex items-center gap-4">
            <Link to="/login" className="font-semibold text-primary-600 hover:text-primary-700 dark:text-primary-300 dark:hover:text-primary-200">
              Đăng nhập
            </Link>
            <Link to="/register" className="font-semibold text-primary-600 hover:text-primary-700 dark:text-primary-300 dark:hover:text-primary-200">
              Đăng ký
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
