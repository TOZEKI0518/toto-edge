type PredictionBarProps = {
  label: string;
  value: number;
};

export function PredictionBar({ label, value }: PredictionBarProps) {
  return (
    <div className="mt-5">
      <div className="mb-2 flex justify-between text-sm">
        <span className="text-white/60">{label}</span>
        <span>{value}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full bg-cyan-300"
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}