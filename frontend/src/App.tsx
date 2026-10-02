import { Navigate, Route, Routes } from "react-router-dom";
import Header from "./components/layout/Header";
import AboutPage from "./pages/AboutPage";
import AnalyzePage from "./pages/AnalyzePage";
import HistoryPage from "./pages/HistoryPage";

export default function App() {
  return (
    <div className="min-h-screen bg-base">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:border focus:border-edge focus:bg-raised focus:px-4 focus:py-2 focus:text-sm focus:text-ink"
      >
        Skip to content
      </a>
      <Header />
      <main id="main-content">
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
