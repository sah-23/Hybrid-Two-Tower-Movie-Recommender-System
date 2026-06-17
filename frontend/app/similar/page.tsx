'use client';
import { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { getSimilarTMDB, SimilarMoviesResult, Movie, searchMoviesTMDB } from '@/lib/api';

export default function SimilarPage() {
  const router = useRouter();

  // ── Search autocomplete state ─────────────────────────────────────────────
  const [query,       setQuery]       = useState('');
  const [suggestions, setSuggestions] = useState<Movie[]>([]);
  const [showDrop,    setShowDrop]    = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Results state ─────────────────────────────────────────────────────────
  const [result,  setResult]  = useState<SimilarMoviesResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');

  // Live autocomplete — fires 250 ms after user stops typing
  useEffect(() => {
    if (query.length < 2) { setSuggestions([]); setShowDrop(false); return; }
    const t = setTimeout(async () => {
      try {
        const movies = await searchMoviesTMDB(query);
        setSuggestions(movies);
        // FIX: Only show dropdown if the input is still actively focused
        if (document.activeElement === inputRef.current) {
          setShowDrop(movies.length > 0);
        }
      } catch {
        setSuggestions([]);
        setShowDrop(false);
      }
    }, 250);
    return () => clearTimeout(t);
  }, [query]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (inputRef.current && !inputRef.current.closest('.search-wrapper')?.contains(e.target as Node)) {
        setShowDrop(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const runSearch = async (title: string) => {
    setQuery(title);
    setSuggestions([]);
    setShowDrop(false);
    inputRef.current?.blur();   // dismisses dropdown and kills the debounce re-trigger
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await getSimilarTMDB(title.trim(), 20);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Movie not found.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 text-white p-4 sm:p-6">
      <div className="max-w-2xl mx-auto">

        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button
            onClick={() => router.push('/rate-new')}
            className="text-gray-400 hover:text-white transition text-xl px-1">
            ←
          </button>
          <div>
            <h1 className="text-2xl font-bold">Find Similar Movies</h1>
            <p className="text-gray-400 text-sm mt-0.5">
              Search for a movie and we'll find content-similar films
            </p>
          </div>
        </div>

        {/* Search with autocomplete */}
        <div className="search-wrapper relative mb-6">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <input
                ref={inputRef}
                className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-3
                           text-white placeholder-gray-500 focus:outline-none focus:border-blue-500
                           focus:ring-1 focus:ring-blue-500 transition-all"
                placeholder="e.g. The Matrix, Inception, Toy Story..."
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && query.trim() && runSearch(query.trim())}
                onFocus={() => suggestions.length > 0 && setShowDrop(true)}
                autoComplete="off"
              />

              {/* Dropdown */}
              {showDrop && suggestions.length > 0 && (
                <div className="absolute z-20 w-full mt-1 bg-gray-800 border border-gray-700
                                rounded-xl max-h-64 overflow-y-auto shadow-2xl">
                  {suggestions.slice(0, 8).map(m => (
                    <button
                      key={m.movie_id}
                      onMouseDown={e => e.preventDefault()}  // prevent input blur before click
                      onClick={() => runSearch(m.title)}
                      className="w-full text-left px-4 py-3 hover:bg-gray-700 transition
                                 border-b border-gray-700/50 last:border-0 group"
                    >
                      <div className="flex items-center gap-3">
                        <span className="text-lg">🎬</span>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-white text-sm truncate">{m.title}</p>
                          <div className="flex items-center gap-2 mt-0.5">
                            {m.year && (
                              <span className="text-xs text-gray-400 bg-gray-700/60 px-1.5 py-0.5 rounded">
                                {m.year}
                              </span>
                            )}
                            <span className="text-xs text-gray-500 truncate">{m.genres}</span>
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={() => query.trim() && runSearch(query.trim())}
              disabled={!query.trim() || loading}
              className="px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                         disabled:text-gray-500 rounded-xl font-semibold transition shrink-0">
              {loading ? '···' : 'Search'}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-500/40 rounded-xl px-4 py-3 mb-4
                          flex items-center gap-2">
            <span>⚠️</span>
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        )}

        {/* Results */}
        {result && (
          <>
            <div className="bg-gray-900 rounded-xl px-4 py-3 mb-4 border border-blue-500/30">
              <p className="text-xs text-gray-400 mb-0.5">Showing movies similar to</p>
              <p className="font-semibold">{result.query.title}</p>
              <p className="text-xs text-gray-400">{result.query.genres}</p>
            </div>

            <div className="space-y-2">
              {result.similar.map((m, i) => (
                <div
                  key={m.movie_id}
                  className="bg-gray-900 rounded-xl px-4 py-3 flex items-center gap-4">
                  <span className="text-gray-600 text-sm w-5 shrink-0 text-right">{i + 1}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-2">
                      <p className="font-medium truncate">{m.title}</p>
                      {m.year && (
                        <span className="text-xs text-gray-500 shrink-0">{m.year}</span>
                      )}
                    </div>
                    <p className="text-xs text-gray-400">{m.genres}</p>
                  </div>
                  <div className="text-right shrink-0">
                    <p className="text-xs text-gray-500">similarity</p>
                    <p className="text-blue-400 font-bold text-sm">
                      {(m.similarity * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Empty state */}
        {!result && !loading && !error && (
          <div className="text-center py-16 text-gray-600">
            <div className="text-5xl mb-3">🎬</div>
            <p className="text-sm">Search for any movie to find similar ones</p>
          </div>
        )}

      </div>
    </main>
  );
}
