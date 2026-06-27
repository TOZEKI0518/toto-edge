import { getTotoLivePageSummary } from "@/services/totoLiveService";

export default async function DebugPage() {
  const summary = await getTotoLivePageSummary();

  return (
    <main className="min-h-screen bg-[#05060A] p-8 text-white">
      <div className="mx-auto max-w-3xl rounded-3xl border border-white/10 bg-white/[0.04] p-6">
        <p className="text-sm text-cyan-300">Debug</p>
        <h1 className="mt-2 text-3xl font-bold">TOTO Live Fetch Test</h1>

        <div className="mt-6 space-y-3 text-white/80">
          <p>
            <span className="text-white/50">Title:</span> {summary.title}
          </p>
          <p>
            <span className="text-white/50">HTML Length:</span>{" "}
            {summary.textLength.toLocaleString()}
          </p>
        </div>
      </div>
    </main>
  );
}