"use client";

import { useState } from "react";

const navItems = [
  { href: "/", label: "Dashboard" },
  { href: "/backtest", label: "Backtest" },
  { href: "/backtest/summary", label: "Summary" },
  { href: "/backtest/inspect", label: "Inspect" },
  { href: "/runs", label: "AI Lab" },
  { href: "/debug", label: "Debug" },
];

const pageHelp = [
  ["Dashboard", "最新回のv6予測、市場との差、購入判断、推奨口を確認します。"],
  ["Backtest", "選択した過去回を、当時までのデータだけで再予測します。"],
  ["Summary", "複数開催回の平均的中率と10口バックテストを比較します。"],
  ["Inspect", "開催回の取得件数、照合率、データ欠損を点検します。"],
  ["AI Lab", "保存済み予測Runや学習・推論処理の履歴を確認します。"],
  ["Debug", "外部サイト取得やパーサーの状態を開発者向けに確認します。"],
];

const metricHelp = [
  ["Hit Rate", "各試合で最上位予測が実結果と一致した割合です。"],
  ["Best Hits", "おすすめ50口のうち、最も実結果に近かった1口の的中数です。"],
  ["1等/2等/3等相当", "13試合開催で13、12、11試合一致した口数です。"],
  ["Edge", "AI確率－市場投票率。プラスが大きいほど市場との乖離があります。"],
  ["Value Ratio", "AI確率÷市場投票率。1を超えるとAIが市場より高く評価しています。"],
  ["Confidence", "確率差、モデル一致度、不確実性をまとめた信頼度です。"],
  ["Coverage", "Single・Double・Tripleの推奨カバー範囲です。"],
  ["Value Index", "過大評価を抑える補正後の、購入判断用の相対指標です。"],
];

export function AppNav() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <div className="mt-6 flex items-start justify-between gap-3">
        <nav className="flex flex-wrap gap-2">
          {navItems.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="inline-flex rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-sm text-white/70 hover:border-cyan-300/40 hover:text-cyan-300"
            >
              {item.label}
            </a>
          ))}
        </nav>

        <button
          type="button"
          onClick={() => setIsOpen(true)}
          aria-label="画面説明を開く"
          className="shrink-0 rounded-2xl border border-white/10 bg-white/[0.05] p-3 text-white/75 hover:border-cyan-300/40 hover:text-cyan-200"
        >
          <span className="block h-0.5 w-6 bg-current" />
          <span className="mt-1.5 block h-0.5 w-6 bg-current" />
          <span className="mt-1.5 block h-0.5 w-6 bg-current" />
        </button>
      </div>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/70 backdrop-blur-sm">
          <button
            type="button"
            aria-label="画面説明を閉じる"
            className="absolute inset-0 cursor-default"
            onClick={() => setIsOpen(false)}
          />

          <aside className="relative h-full w-full max-w-md overflow-y-auto border-l border-white/10 bg-[#0B0D13] p-6 shadow-2xl">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-cyan-300">TOTO EDGE Guide</p>
                <h2 className="mt-1 text-2xl font-black">画面・指標の使い方</h2>
              </div>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="rounded-full border border-white/10 px-3 py-2 text-white/60"
              >
                ✕
              </button>
            </div>

            <section className="mt-7">
              <h3 className="font-bold text-white">各画面</h3>
              <div className="mt-3 space-y-3">
                {pageHelp.map(([title, description]) => (
                  <div
                    key={title}
                    className="rounded-2xl border border-white/10 bg-white/[0.035] p-4"
                  >
                    <p className="font-bold text-cyan-200">{title}</p>
                    <p className="mt-1 text-sm leading-6 text-white/55">
                      {description}
                    </p>
                  </div>
                ))}
              </div>
            </section>

            <section className="mt-7">
              <h3 className="font-bold text-white">主要指標</h3>
              <div className="mt-3 space-y-3">
                {metricHelp.map(([title, description]) => (
                  <div
                    key={title}
                    className="rounded-2xl border border-white/10 bg-white/[0.035] p-4"
                  >
                    <p className="font-bold text-emerald-200">{title}</p>
                    <p className="mt-1 text-sm leading-6 text-white/55">
                      {description}
                    </p>
                  </div>
                ))}
              </div>
            </section>

            <p className="mt-7 rounded-2xl border border-amber-300/20 bg-amber-300/[0.06] p-4 text-sm leading-6 text-amber-100/75">
              的中率が高くても利益が出るとは限りません。購入判断では、
              Edge・Value・おすすめ50口の最高一致数を併せて確認します。
            </p>
          </aside>
        </div>
      )}
    </>
  );
}
