export type Outcome = 'home' | 'draw' | 'away'

export type MatchInput = {
  id: number
  round: string
  kickoff: string
  homeTeam: string
  awayTeam: string
  venue: string
  homeRank: number
  awayRank: number
  homeRecentPoints: number
  awayRecentPoints: number
  homeGoalDiff: number
  awayGoalDiff: number
  homeHomeStrength: number
  awayAwayStrength: number
  homeRestDays: number
  awayRestDays: number
  weatherRisk: number
  homeInjuryRisk: number
  awayInjuryRisk: number
  motivationHome: number
  motivationAway: number
  publicVote: Record<Outcome, number>
}

export type Reason = {
  label: string
  value: number
  type: 'plus' | 'minus' | 'neutral'
}

export type Prediction = MatchInput & {
  scores: Record<Outcome, number>
  probabilities: Record<Outcome, number>
  pick: Outcome
  confidence: 'S' | 'A' | 'B' | 'C'
  edge: number
  reasons: Reason[]
}
