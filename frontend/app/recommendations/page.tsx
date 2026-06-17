'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Recommendation } from '@/lib/api';

function ScoreBar({ rating, min, max }: {
  rating: number; min: number; max: number;
}) {
  const range = max - min || 0.001;
  const pct   = ((rating - min) / range) * 75 + 25; // 25%–100%
  return (
    <div className="flex items-center gap-2 mt-2">
      <div className="flex-1 bg-gray-700 rounded-full h-1.5">
        <div
          className="bg-yellow-400 h-1.5 rounded-full"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-yellow-400 text-xs font-bold w-12 text-right tabular-nums">
        {rating.toFixed(4)}
      </span>
    </div>
  );
}

export default function RecommendationsPage() {
  const router = useRouter();
  const [recs, setRecs] = useState<Recommendation[]>([]);

  useEffect(() => {
    const raw = sessionStorage.getItem('recommendations');
    if (!raw) { router.push('/rate'); return; }
    setRecs(JSON.parse(raw));
  }, [router]);

  if (recs.length === 0) return (
    <main className="min-h-screen flex items-center justify-center bg-gray-950 text-white">
      <p className="animate-pulse">Loading…</p>
    </main>
  );

  const max = Math.max(...recs.map(r => r.predicted_rating));
  const min = Math.min(...recs.map(r => r.predicted_rating));

  return (
    <main className="min-h-screen bg-gray-950 text-white p-6">
      <div className="max-w-2xl mx-auto">

        <div className="flex items-center justify-between mb-2">
          <h1 className="text-2xl font-bold">Your Recommendations</h1>
          <button onClick={() => router.push('/rate-new')}
            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-xl text-sm transition">
            ← Rate again
          </button>
        </div>

        <p className="text-gray-500 text-xs mb-6">
          Scores range from{' '}
          <span className="text-yellow-400 font-mono">{min.toFixed(4)}</span>
          {' '}to{' '}
          <span className="text-yellow-400 font-mono">{max.toFixed(4)}</span>
          {' '}· bar shows relative preference within this list
        </p>

        <div className="space-y-3">
          {recs.map((rec, i) => (
            <div key={rec.movie_id}
              className="bg-gray-900 rounded-xl p-4 flex items-center gap-4">
              <span className="text-xl font-bold text-gray-600 w-7 text-right shrink-0">
                {i + 1}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2">
                  <p className="font-semibold truncate">{rec.title}</p>
                  {rec.year && (
                    <span className="text-xs text-gray-500 shrink-0">{rec.year}</span>
                  )}
                </div>
                <p className="text-xs text-gray-400 mt-0.5">
                  {rec.genres.replace(/\|/g, ' · ')}
                </p>
                <ScoreBar rating={rec.predicted_rating} min={min} max={max} />
              </div>
            </div>
          ))}
        </div>

      </div>
    </main>
  );
}