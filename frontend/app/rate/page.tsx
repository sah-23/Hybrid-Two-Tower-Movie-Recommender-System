'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Movie, RatedMovie, Demographics, searchMovies, getRecommendations } from '@/lib/api';

const STARS = [1, 2, 3, 4, 5];

const AGE_OPTIONS = [
  { label: 'Under 18', value: 1  },
  { label: '18–24',    value: 18 },
  { label: '25–34',    value: 25 },
  { label: '35–44',    value: 35 },
  { label: '45–49',    value: 45 },
  { label: '50–55',    value: 50 },
  { label: '56+',      value: 56 },
];

const OCCUPATION_OPTIONS = [
  [0,  'Other / not listed'],
  [1,  'Academic / Educator'],
  [2,  'Artist'],
  [3,  'Clerical / Admin'],
  [4,  'College / Grad Student'],
  [5,  'Customer Service'],
  [6,  'Doctor / Health Care'],
  [7,  'Executive / Managerial'],
  [8,  'Farmer'],
  [9,  'Homemaker'],
  [10, 'K-12 Student'],
  [11, 'Lawyer'],
  [12, 'Programmer'],
  [13, 'Retired'],
  [14, 'Sales / Marketing'],
  [15, 'Scientist'],
  [16, 'Self-employed'],
  [17, 'Technician / Engineer'],
  [18, 'Tradesman / Craftsman'],
  [19, 'Unemployed'],
  [20, 'Writer'],
] as [number, string][];

export default function RatePage() {
  const router = useRouter();

  // ── Demographics ─────────────────────────────────────────────────────────
  const [demo, setDemo] = useState<Demographics>({
    gender: 'M', age: 25, occupation: 4,
  });

  // ── Movie search & rating ─────────────────────────────────────────────────
  const [query,   setQuery]   = useState('');
  const [results, setResults] = useState<Movie[]>([]);
  const [rated,   setRated]   = useState<(RatedMovie & { title: string })[]>([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');
  const [showUnderConstruction, setShowUnderConstruction] = useState(false);
  // Debounced search
  useEffect(() => {
    if (query.length < 2) { setResults([]); return; }
    const t = setTimeout(async () => {
      const movies  = await searchMovies(query);
      const ratedIds = new Set(rated.map(r => r.movie_id));
      setResults(movies.filter(m => !ratedIds.has(m.movie_id)));
    }, 300);
    return () => clearTimeout(t);
  }, [query, rated]);

  const addMovie = (movie: Movie) => {
    if (rated.length >= 15 || rated.some(r => r.movie_id === movie.movie_id)) return;
    setRated(prev => [...prev, {
      movie_id: movie.movie_id,
      title   : movie.title,
      rating  : 3,
      order   : prev.length,
    }]);
    setQuery('');
    setResults([]);
  };

  const setRating = (movie_id: number, rating: number) => {
    setRated(prev => prev.map(r => r.movie_id === movie_id ? { ...r, rating } : r));
  };

  const removeMovie = (movie_id: number) => {
    setRated(prev =>
      prev.filter(r => r.movie_id !== movie_id)
          .map((r, i) => ({ ...r, order: i }))
    );
  };

  // ── Submit ────────────────────────────────────────────────────────────────
  const submit = async () => {
    if (rated.length !== 15) return;
    setLoading(true);
    setError('');
    try {
      const ratings = rated.map(({ movie_id, rating, order }) => ({
        movie_id, rating, order,
      }));
      const recs = await getRecommendations(demo, ratings);
      sessionStorage.setItem('recommendations', JSON.stringify(recs));
      router.push('/recommendations');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Something went wrong.');
      setLoading(false);
    }
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 text-white p-3 sm:p-4 lg:p-5">
      <div className="max-w-3xl mx-auto space-y-5 py-4">

        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-block">
            <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent mb-1.5">
              🎬 Movie Recommender (OLDER MOVIES)

            </h1>
          <div className="h-0.5 bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 rounded-full"></div>
        </div>
          <p className="text-gray-400 text-sm leading-relaxed whitespace-nowrap">
            Tell us about yourself, rate 15 movies you've seen, and we'll recommend your next favorite film
            using MovieLense 1M
          </p>

        </div>

        {/* Demographics */}
        <section className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50 shadow-xl">
          <h2 className="font-semibold text-base text-gray-100 mb-3 flex items-center gap-2">
            <span className="text-lg">👤</span> About You
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">

            <div>
              <label className="text-xs font-medium text-gray-400 block mb-1.5">Gender</label>
              <select
                value={demo.gender}
                onChange={e => setDemo(d => ({ ...d, gender: e.target.value as 'M' | 'F' }))}
                className="w-full bg-gray-900/80 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              >
                <option value="M">Male</option>
                <option value="F">Female</option>
              </select>
            </div>

            <div>
              <label className="text-xs font-medium text-gray-400 block mb-1.5">Age Group</label>
              <select
                value={demo.age}
                onChange={e => setDemo(d => ({ ...d, age: Number(e.target.value) }))}
                className="w-full bg-gray-900/80 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              >
                {AGE_OPTIONS.map(o => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs font-medium text-gray-400 block mb-1.5">Occupation</label>
              <select
                value={demo.occupation}
                onChange={e => setDemo(d => ({ ...d, occupation: Number(e.target.value) }))}
                className="w-full bg-gray-900/80 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              >
                {OCCUPATION_OPTIONS.map(([v, l]) => (
                  <option key={v} value={v}>{l}</option>
                ))}
              </select>
            </div>

          </div>
        </section>

        {/* Movie Rating Section */}
        <section className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50 shadow-xl">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-base text-gray-100 flex items-center gap-2">
              <span className="text-lg">⭐</span> Rate 15 Movies
            </h2>
            <div className={`px-3 py-1 rounded-full font-semibold text-xs ${
              rated.length === 15 
                ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                : 'bg-teal-500/20 text-teal-400 border border-teal-500/30'
            }`}>
              {rated.length} / 15
            </div>
          </div>

          {/* Search Input */}
          {rated.length < 15 && (
            <div className="relative mb-3">
              <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 text-lg">
                🔍
              </div>
              <input
                className="w-full bg-gray-900/80 border border-gray-600 rounded-lg pl-10 pr-3 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent transition-all"
                placeholder="Search for a movie by title…"
                value={query}
                onChange={e => setQuery(e.target.value)}
                autoComplete="off"
              />
              {results.length > 0 && (
                <div className="absolute z-10 w-full mt-1.5 bg-gray-800 border border-gray-600 rounded-lg max-h-60 overflow-y-auto shadow-2xl">
                  {results.slice(0, 8).map(m => (
                    <button
                      key={m.movie_id}
                      onClick={() => addMovie(m)}
                      className="w-full text-left px-3 py-2.5 hover:bg-gray-700/70 transition-colors border-b border-gray-700 last:border-0 group"
                    >
                      <div className="flex items-start gap-2">
                        <span className="text-lg group-hover:scale-110 transition-transform">🎬</span>
                        <div className="flex-1 min-w-0">
                          <span className="font-medium text-white text-sm block">{m.title}</span>
                          <div className="flex items-center gap-2 mt-0.5">
                            {m.year && <span className="text-xs text-gray-400 bg-gray-700/50 px-1.5 py-0.5 rounded">{m.year}</span>}
                            <span className="text-xs text-gray-400">{m.genres}</span>
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Rated Movies List */}
          <div className="space-y-2 max-h-[45vh] overflow-y-auto pr-1">
            {rated.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <div className="text-3xl mb-2">🎥</div>
                <p className="text-xs">Start by searching for movies you've watched</p>
              </div>
            )}
            {rated.map((r, i) => (
              <div key={r.movie_id}
                className="bg-gray-900/60 border border-gray-700/50 rounded-lg px-3 py-2.5 flex items-center gap-3 hover:border-gray-600 transition-all group">
                <span className="text-gray-500 font-semibold text-xs w-5 text-center shrink-0 bg-gray-800/50 rounded py-0.5">
                  {i + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-white text-sm truncate mb-1.5">{r.title}</p>
                  <div className="flex gap-1">
                    {STARS.map(s => (
                      <button key={s} onClick={() => setRating(r.movie_id, s)}
                        className={`text-xl transition-all hover:scale-110 ${
                          s <= r.rating ? 'text-yellow-400 drop-shadow-[0_0_6px_rgba(250,204,21,0.4)]' : 'text-gray-700 hover:text-gray-600'
                        }`}>
                        ★
                      </button>
                    ))}
                  </div>
                </div>
                <button onClick={() => removeMovie(r.movie_id)}
                  className="text-gray-600 hover:text-red-400 hover:bg-red-400/10 transition-all shrink-0 text-lg w-7 h-7 rounded-lg flex items-center justify-center">
                  ✕
                </button>
              </div>
            ))}
          </div>
        </section>

        {/* Error Message */}
        {error && (
          <div className="bg-red-900/30 border border-red-500/50 rounded-lg px-4 py-3 flex items-start gap-2">
            <span className="text-lg">⚠️</span>
            <p className="text-red-300 text-xs flex-1">{error}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          onClick={submit}
          disabled={rated.length !== 15 || loading}
          className="w-full py-3.5 bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:via-teal-500 hover:to-cyan-500
                              disabled:from-gray-800 disabled:via-gray-800 disabled:to-gray-800 disabled:text-gray-400 disabled:cursor-not-allowed
                              rounded-lg font-semibold text-base transition-all shadow-lg hover:shadow-xl hover:scale-[1.01]
                              disabled:hover:scale-100 disabled:shadow-none active:scale-[0.99]"
        >
          {loading
            ? '✨ Getting recommendations…'
            : rated.length === 15
              ? '🎯 Get My Recommendations →'
              : `📝 Add ${15 - rated.length} or more movie${15 - rated.length !== 1 ? 's' : ''} to continue`}
        </button>






        {/* Under Construction Modal */}
        {showUnderConstruction && (
          <div
            className="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
            onClick={() => setShowUnderConstruction(false)}
          >
            <div
              className="bg-gray-900 border border-gray-700 rounded-2xl p-8 max-w-sm mx-4 text-center"
              onClick={e => e.stopPropagation()}
            >
              <div className="text-5xl mb-4">🚧</div>
              <h2 className="text-xl font-bold mb-2">Under Construction</h2>
              <p className="text-gray-400 text-sm mb-6">
                Newer movie recommendations are coming soon.
              </p>
              <button
                onClick={() => setShowUnderConstruction(false)}
                className="px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded-xl transition text-sm"
              >
                ← Back
              </button>
            </div>
          </div>
        )}

        {/* Bottom buttons */}
        <div className="flex flex-col items-center gap-3 pb-6">

          <button
            onClick={() => router.push('/rate-new')}
            className="w-full py-3 bg-transparent border border-gray-700 hover:border-gray-500 hover:bg-gray-800/50 rounded-lg text-gray-400 hover:text-white text-sm font-medium transition-all">
            ← Back to the Main Recommender
          </button>


          {/* Back */}

          </div>

      </div>
    </main>
  );
}
