import { Prediction } from '@/lib/types'
import { labelOutcome } from '@/lib/predict'

export function DetailPanel({ match }: { match: Prediction }) {
  return (
    <div className="card">
      <div className="cardHeader">
        <div>
          <h2>{match.homeTeam} vs {match.awayTeam}</h2>
          <span>推奨：{labelOutcome(match.pick)} / 信頼度 {match.confidence}</span>
        </div>
        <div className="badge">期待値 {match.edge > 0 ? '+' : ''}{match.edge}%</div>
      </div>
      <div style={{ padding: 20 }}>
        {match.reasons.map((reason) => (
          <div className="reason" key={reason.label}>
            <span>{reason.label}</span>
            <b className={reason.type === 'plus' ? 'plus' : reason.type === 'minus' ? 'minus' : ''}>
              {reason.value > 0 ? '+' : ''}{reason.value}
            </b>
          </div>
        ))}
        <div className="footerNote">
          ※ MVPではサンプルデータを使ったルールベース予想です。次ステップで実データ取得API、開催回登録、結果保存、バックテストに接続します。
        </div>
      </div>
    </div>
  )
}
