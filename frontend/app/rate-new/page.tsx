'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Movie, RatedMovie, searchMovies25m, getRecommendations25m } from '@/lib/api';

const STARS = [1, 2, 3, 4, 5];

export default function RateNewPage() {
  const router = useRouter();

  const [query,   setQuery]   = useState('');
  const [results, setResults] = useState<Movie[]>([]);
  const [rated,   setRated]   = useState<(RatedMovie & { title: string })[]>([]);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState('');

  useEffect(() => {
    if (query.length < 2) { setResults([]); return; }
    const t = setTimeout(async () => {
      try {
        const movies   = await searchMovies25m(query);
        const ratedIds = new Set(rated.map(r => r.movie_id));
        setResults(movies.filter(m => !ratedIds.has(m.movie_id)));
      } catch {
        setResults([]);
      }
    }, 250);
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

  const submit = async () => {
    if (rated.length !== 15) return;
    setLoading(true);
    setError('');
    try {
      const ratings = rated.map(({ movie_id, rating, order }) => ({
        movie_id, rating, order,
      }));
      const recs = await getRecommendations25m(ratings);
      sessionStorage.setItem('recommendations', JSON.stringify(recs));
      router.push('/recommendations');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Something went wrong.');
      setLoading(false);
    }
  };

  const progress = (rated.length / 15) * 100;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Instrument+Sans:wght@300;400;500;600&display=swap');

        :root {
          --bg:        #0a0806;
          --bg-card:   #110e0b;
          --bg-row:    rgba(255,255,255,0.06);
          --border:    rgba(255,200,80,0.14);
          --border-hi: rgba(255,200,80,0.38);
          --gold:      #f0c040;
          --gold-glow: rgba(240,192,64,0.6);
          --gold-dim:  rgba(240,192,64,0.8);
          --red:       #d94030;
          --red-glow:  rgba(217,64,48,0.5);
          --cream:     #faf2e4;
          --muted:     #cba882;
          --dim:       #8a7060;
          --star-on:   #f0c040;
          --star-off:  rgba(255,255,255,0.32);
        }

        /* ── Animated orbs ── */
        @keyframes driftA {
          0%   { transform: translate(0, 0) scale(1); }
          33%  { transform: translate(40px, -30px) scale(1.08); }
          66%  { transform: translate(-20px, 20px) scale(0.95); }
          100% { transform: translate(0, 0) scale(1); }
        }
        @keyframes driftB {
          0%   { transform: translate(0, 0) scale(1); }
          40%  { transform: translate(-50px, 30px) scale(1.1); }
          70%  { transform: translate(30px, -20px) scale(0.92); }
          100% { transform: translate(0, 0) scale(1); }
        }
        @keyframes driftC {
          0%   { transform: translate(0, 0) scale(1); }
          50%  { transform: translate(25px, 40px) scale(1.06); }
          100% { transform: translate(0, 0) scale(1); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.7; }
          50%       { opacity: 1; }
        }

        /* ── Left breathing blob ── */
        @keyframes breatheLeft {
          0%   { transform: scale(1)    translateY(0);   opacity: 0.55; }
          40%  { transform: scale(1.12) translateY(-28px); opacity: 0.85; }
          70%  { transform: scale(0.94) translateY(18px);  opacity: 0.65; }
          100% { transform: scale(1)    translateY(0);   opacity: 0.55; }
        }

        .left-blob {
          position: fixed;
          left: -260px;
          top: 15%;
          width: 480px;
          height: 480px;
          border-radius: 50%;
          background: radial-gradient(
            circle,
            rgba(200, 60, 30, 0.55) 0%,
            rgba(210, 80, 20, 0.28) 40%,
            transparent 70%
          );
          filter: blur(70px);
          pointer-events: none;
          z-index: 0;
          animation: breatheLeft 13s ease-in-out infinite -3s;
        }

        /* ── Quiet Center Glare ── */
        @keyframes driftCenterQuiet {
          0%   { transform: translate(0, 0) scale(1); opacity: 0.08; }
          50%  { transform: translate(4vw, 5vh) scale(1.1); opacity: 0.15; }
          100% { transform: translate(0, 0) scale(1); opacity: 0.08; }
        }

        .center-glare {
          position: fixed;
          top: 20%;
          left: 25%;
          width: 50vw;
          height: 50vh;
          background: radial-gradient(circle, var(--red) 0%, var(--gold) 40%, transparent 70%);
          filter: blur(100px);
          pointer-events: none;
          z-index: 0;
          animation: driftCenterQuiet 28s ease-in-out infinite alternate;
        }


        /* ── Side ambient gradient ── */
        @keyframes ambientLeft {
          0%   { opacity: 0.55; transform: scale(1) translateY(0); }
          50%  { opacity: 0.9;  transform: scale(1.08) translateY(-30px); }
          100% { opacity: 0.55; transform: scale(1) translateY(0); }
        }
        @keyframes ambientRight {
          0%   { opacity: 0.4;  transform: scale(1) translateY(0); }
          50%  { opacity: 0.72; transform: scale(1.05) translateY(20px); }
          100% { opacity: 0.4;  transform: scale(1) translateY(0); }
        }

        .ambient-left {
          position: fixed;
          left: -180px;
          top: 5%;
          width: 500px;
          height: 90vh;
          background: radial-gradient(
            ellipse at left center,
            rgba(180, 50, 20, 0.38) 0%,
            rgba(220, 80, 30, 0.18) 35%,
            transparent 70%
          );
          pointer-events: none;
          z-index: 0;
          filter: blur(40px);
          animation: ambientLeft 11s ease-in-out infinite;
        }

        .ambient-right {
          position: fixed;
          right: -160px;
          top: 10%;
          width: 440px;
          height: 80vh;
          background: radial-gradient(
            ellipse at right center,
            rgba(200, 130, 20, 0.35) 0%,
            rgba(240, 160, 40, 0.25) 40%,
            transparent 70%
          );
          pointer-events: none;
          z-index: 0;
          filter: blur(50px);
          animation: ambientRight 15s ease-in-out infinite -6s;
        }

        .orb {
          position: fixed;
          border-radius: 50%;
          filter: blur(80px);
          pointer-events: none;
          z-index: 0;
        }
        .orb-1 {
          width: 520px; height: 520px;
          top: -160px; left: -100px;
          background: radial-gradient(circle, rgba(217,64,48,0.28) 0%, transparent 70%);
          animation: driftA 14s ease-in-out infinite, pulse 8s ease-in-out infinite;
        } 
        .orb-2 {
          width: 480px; height: 480px;
          top: -80px; right: -120px;
          background: radial-gradient(circle, rgba(240,192,64,0.35) 0%, transparent 70%);
          animation: driftB 18s ease-in-out infinite -4s, pulse 10s ease-in-out infinite -3s;
        }
        .orb-3 {
          width: 400px; height: 400px;
          bottom: -100px; left: 50%;
          transform: translateX(-50%);
          background: radial-gradient(circle, rgba(180,80,20,0.2) 0%, transparent 70%);
          animation: driftC 20s ease-in-out infinite;
        }
        .orb-4 {
          width: 300px; height: 300px;
          bottom: 20%; right: -80px;
          background: radial-gradient(circle, rgba(240,140,40,0.15) 0%, transparent 70%);
          animation: driftA 22s ease-in-out infinite reverse;
        }

        /* grain overlay */
        .page-root::after {
          content: '';
          position: fixed; inset: 0;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
          opacity: 0.5;
          pointer-events: none;
          z-index: 100;
        }

        .page-root {
          height: 100dvh;
          overflow: hidden;
          background: var(--bg);
          font-family: 'Instrument Sans', sans-serif;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: 0 1rem;
          position: relative;
        }

        .inner {
          width: 100%;
          max-width: 640px;
          height: 100%;
          position: relative;
          z-index: 1;
          display: flex;
          flex-direction: column;
          /* push content down from the very top */
          padding: 1.6rem 0 0.5rem;
        }

        /* ── HERO ── */
        .hero {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: 0 0.25rem 0.75rem;
          flex-shrink: 0;
          border-bottom: 1px solid rgba(255,200,80,0.1);
          margin-bottom: 0.7rem;
        }

        .hero-title {
          font-family: 'Playfair Display', serif;
          font-size: clamp(1.4rem, 4vw, 2rem);
          font-weight: 900;
          letter-spacing: -0.01em;
          line-height: 1.1;
          color: var(--cream);
          /* more breathing room before the subtitle */
          margin: 0 0 0.55rem;
          text-shadow: 0 0 40px rgba(240,192,64,0.15);
        }

        .hero-title em {
          font-style: italic;
          color: var(--gold);
          text-shadow: 0 0 30px rgba(240,192,64,0.45);
        }

        .hero-sub {
          font-size: 0.77rem;
          font-weight: 400;
          color: var(--muted);
          line-height: 1.5;
          margin: 0;
        }

        /* ── PROGRESS ── */
        .progress-wrap {
          display: flex;
          align-items: center;
          gap: 0.85rem;
          /* ③ tighter */
          margin-bottom: 0.55rem;
          flex-shrink: 0;
        }

        .progress-track {
          flex: 1;
          height: 2px;
          background: rgba(255,200,80,0.12);
          border-radius: 100px;
          overflow: visible;
          position: relative;
        }

        .progress-fill {
          height: 2px;
          border-radius: 100px;
          background: linear-gradient(90deg, var(--red), var(--gold));
          transition: width 0.5s cubic-bezier(0.34,1.56,0.64,1);
          position: relative;
          box-shadow: 0 0 8px rgba(240,192,64,0.5);
        }

        .progress-fill::after {
          content: '';
          position: absolute;
          right: -4px; top: -4px;
          width: 9px; height: 9px;
          border-radius: 50%;
          background: var(--gold);
          box-shadow: 0 0 12px var(--gold), 0 0 24px rgba(240,192,64,0.4);
          opacity: var(--dot-opacity, 0);
          transition: opacity 0.3s;
        }

        .progress-label {
          font-size: 0.68rem;
          font-weight: 600;
          letter-spacing: 0.1em;
          color: var(--dim);
          font-variant-numeric: tabular-nums;
          min-width: 2.6rem;
          text-align: right;
          transition: color 0.3s;
        }
        .progress-label.done { color: var(--gold); text-shadow: 0 0 12px rgba(240,192,64,0.5); }

        /* ── CARD ── */
        .card {
          flex: 1;
          min-height: 0;
          /* ④ cap the card so it doesn't stretch to full remaining height */
          max-height: 72vh;
          background: rgba(17,14,11,0.85);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(255,200,80,0.12);
          border-radius: 18px;
          overflow: hidden;
          box-shadow:
            0 0 0 1px rgba(0,0,0,0.5),
            0 32px 80px rgba(0,0,0,0.6),
            inset 0 1px 0 rgba(255,200,80,0.09);
          display: flex;
          flex-direction: column;
        }

        /* ── Search ── */
        .search-wrap {
          padding: 0.75rem 0.9rem 0.7rem;
          border-bottom: 1px solid rgba(255,200,80,0.08);
          flex-shrink: 0;
          position: relative;
          overflow: visible;
          z-index: 10;
        }

        .search-field {
          position: relative;
          display: flex;
          align-items: center;
        }

        .search-icon {
          position: absolute;
          left: 0.85rem;
          top: 50%;
          transform: translateY(-50%);
          font-size: 0.78rem;
          pointer-events: none;
          opacity: 0.6;
        }

        .search-input {
          width: 100%;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,200,80,0.18);
          border-radius: 10px;
          padding: 0.68rem 1rem 0.68rem 2.35rem;
          font-size: 0.855rem;
          color: var(--cream);
          font-family: 'Instrument Sans', sans-serif;
          font-weight: 400;
          transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
          box-sizing: border-box;
        }

        .search-input:focus {
          outline: none;
          border-color: rgba(240,192,64,0.5);
          background: rgba(240,192,64,0.06);
          box-shadow: 0 0 0 3px rgba(240,192,64,0.1), 0 0 20px rgba(240,192,64,0.09);
        }

        /* Brighter placeholder text as requested */
        .search-input::placeholder { 
          color: rgba(255,255,255,0.9); 
          font-style: normal; 
          font-weight: 500;
        }

        /* Dropdown */
        .dropdown {
          position: absolute;
          z-index: 50;
          top: calc(100% + 6px);
          left: 0; right: 0;
          background: #1c1510;
          border: 1px solid rgba(240,192,64,0.25);
          border-radius: 13px;
          overflow-y: auto;
          max-height: 280px;
          box-shadow: 0 24px 70px rgba(0,0,0,0.75), 0 0 0 1px rgba(0,0,0,0.5);
        }

        .dropdown::-webkit-scrollbar { width: 3px; }
        .dropdown::-webkit-scrollbar-track { background: transparent; }
        .dropdown::-webkit-scrollbar-thumb { background: rgba(240,192,64,0.25); border-radius: 4px; }

        .dropdown-item {
          width: 100%;
          text-align: left;
          padding: 0.65rem 0.85rem;
          background: none;
          border: none;
          border-bottom: 1px solid rgba(255,200,80,0.07);
          cursor: pointer;
          transition: background 0.15s;
          display: flex;
          align-items: flex-start;
          gap: 0.65rem;
          font-family: 'Instrument Sans', sans-serif;
        }

        .dropdown-item:last-child { border-bottom: none; }
        .dropdown-item:hover { background: rgba(240,192,64,0.09); }

        .dropdown-item-icon {
          width: 30px; height: 30px;
          background: rgba(240,192,64,0.1);
          border: 1px solid rgba(240,192,64,0.2);
          border-radius: 7px;
          display: flex; align-items: center; justify-content: center;
          font-size: 0.8rem; flex-shrink: 0;
        }

        .dropdown-item-title {
          font-size: 0.82rem; font-weight: 500;
          color: var(--cream);
          display: block; margin-bottom: 0.13rem;
        }

        .dropdown-item-meta { display: flex; gap: 0.4rem; align-items: center; }

        .tag {
          font-size: 0.63rem;
          background: rgba(255,200,80,0.1);
          color: var(--muted);
          padding: 0.07rem 0.38rem;
          border-radius: 4px;
          letter-spacing: 0.02em;
        }

        /* ── Movie list ── */
        .movie-list {
          flex: 1;
          min-height: 0;
          overflow-y: auto;
          padding: 0.55rem 0.9rem;
          display: flex;
          flex-direction: column;
          gap: 0.36rem;
        }

        .movie-list::-webkit-scrollbar { width: 3px; }
        .movie-list::-webkit-scrollbar-track { background: transparent; }
        .movie-list::-webkit-scrollbar-thumb { background: rgba(240,192,64,0.22); border-radius: 4px; }

        /* ── Empty state ── */
        .empty-state {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          gap: 0.9rem;
          padding: 2rem 1.5rem;
        }

        .empty-reel {
          font-size: 2rem;
          opacity: 0.25;
          line-height: 1;
        }

        .empty-text {
          font-size: 0.9rem;
          font-weight: 500;
          line-height: 1.7;
          color: var(--cream);
          font-style: normal;
          font-family: 'Instrument Sans', sans-serif;
          letter-spacing: 0.01em;
        }

        /* ── Movie rows ── */
        .movie-row {
          display: flex;
          align-items: center;
          gap: 0.8rem;
          background: var(--bg-row);
          border: 1px solid rgba(255,200,80,0.1);
          border-radius: 10px;
          padding: 0.5rem 0.72rem;
          transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
          animation: slideIn 0.22s ease both;
        }

        @keyframes slideIn {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        .movie-row:hover {
          border-color: rgba(240,192,64,0.28);
          background: rgba(240,192,64,0.05);
          box-shadow: 0 0 16px rgba(240,192,64,0.07);
        }

        .movie-index {
          font-size: 0.62rem; font-weight: 600;
          color: rgba(240,192,64,0.5);
          width: 16px; text-align: center; flex-shrink: 0;
          font-variant-numeric: tabular-nums;
          font-family: 'Playfair Display', serif;
          font-style: italic;
        }

        .movie-info { flex: 1; min-width: 0; }

        .movie-title {
          font-size: 0.87rem; font-weight: 500;
          color: var(--cream);
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          margin-bottom: 0.28rem;
          letter-spacing: 0.01em;
        }

        .stars { display: flex; gap: 3px; }

        .star-btn {
          background: none; border: none; cursor: pointer;
          font-size: 1rem; line-height: 1; padding: 1px;
          transition: transform 0.12s, color 0.1s, text-shadow 0.1s;
          color: var(--star-off);
        }
        .star-btn:hover { transform: scale(1.3); }
        .star-btn.active {
          color: var(--star-on);
          text-shadow: 0 0 8px rgba(240,192,64,0.65);
        }

        .remove-btn {
          background: none; border: none; cursor: pointer;
          color: rgba(255,255,255,0.3);
          font-size: 0.62rem; width: 24px; height: 24px;
          border-radius: 6px;
          display: flex; align-items: center; justify-content: center;
          transition: color 0.15s, background 0.15s; flex-shrink: 0;
        }
        .remove-btn:hover { color: #ff6060; background: rgba(200,60,60,0.14); }

        /* ── Divider ── */
        .card-divider {
          height: 1px;
          background: rgba(255,200,80,0.08);
          margin: 0 0.9rem;
          flex-shrink: 0;
        }

        /* ── Submit ── */
        .submit-area {
          padding: 0.6rem 0.9rem;
          flex-shrink: 0;
        }

        .submit-btn {
          width: 100%;
          padding: 0.78rem;
          border: none; border-radius: 10px;
          font-family: 'Instrument Sans', sans-serif;
          font-size: 0.88rem; font-weight: 600;
          cursor: pointer;
          transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
          letter-spacing: 0.025em;
        }

        .submit-btn.ready {
          background: linear-gradient(135deg, var(--red) 0%, var(--gold) 100%);
          color: #0a0806;
          box-shadow: 0 4px 28px rgba(217,64,48,0.38), 0 0 50px rgba(240,192,64,0.14);
        }
        .submit-btn.ready:hover {
          opacity: 0.93; transform: translateY(-1px);
          box-shadow: 0 8px 40px rgba(240,192,64,0.38);
        }
        .submit-btn.ready:active { transform: translateY(0); }

        .submit-btn.disabled {
          background: rgba(255,255,255,0.06);
          color: rgba(255,242,200,0.55);
          border: 1px solid rgba(255,200,80,0.18);
          cursor: not-allowed;
          font-style: italic;
        }

        /* ── Error ── */
        .error-box {
          background: rgba(200,50,40,0.09);
          border: 1px solid rgba(200,50,40,0.3);
          border-radius: 9px;
          padding: 0.7rem 0.9rem;
          font-size: 0.78rem; color: #ff7060;
          margin: 0 0.9rem 0.5rem;
          display: flex; gap: 0.5rem; align-items: flex-start;
          flex-shrink: 0;
        }

        /* ── Footer ── */
        .footer-links {
          display: flex;
          flex-direction: row;
          align-items: center;
          justify-content: center;
          gap: 0.6rem;
          padding: 0.6rem 0 0;
          flex-shrink: 0;
        }

        .footer-divider {
          width: 1px; height: 14px;
          background: rgba(255,200,80,0.3);
          flex-shrink: 0;
        }

        .link-btn {
          background: none; border: none; cursor: pointer;
          font-family: 'Instrument Sans', sans-serif;
          transition: color 0.2s, text-shadow 0.2s;
          white-space: nowrap;
        }

        .link-primary {
          font-size: 0.88rem; font-weight: 700;
          background: linear-gradient(90deg, var(--gold) 0%, #ffe080 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-decoration: underline;
          text-underline-offset: 4px;
          text-decoration-color: rgba(240,192,64,0.55);
          filter: drop-shadow(0 0 10px rgba(240,192,64,0.5));
        }
        .link-primary:hover {
          filter: drop-shadow(0 0 18px rgba(240,192,64,0.85));
          text-decoration-color: rgba(240,192,64,0.9);
        }

        .link-secondary {
          font-size: 0.88rem; font-weight: 500;
          color: #ffffff;
          text-decoration: underline;
          text-underline-offset: 4px;
          text-decoration-color: rgba(255,255,255,0.4);
          font-style: italic;
          text-shadow: 0 0 12px rgba(255,255,255,0.25);
        }
        .link-secondary:hover {
          color: #ffffff;
          text-shadow: 0 0 20px rgba(255,255,255,0.55);
          text-decoration-color: rgba(255,255,255,0.7);
        }
      `}</style>
      {/* ... (keep your entire <style> block exactly the same) ... */}

      <main className="page-root">
        {/* MOVING GLARE AND AMBIENT EFFECTS GO INSIDE PAGE-ROOT */}
        <div className="left-blob" />
                {/* NEW: Quiet organic center glare */}
        <div className="center-glare" />
        <div className="ambient-left" />
        <div className="ambient-right" />
        
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="orb orb-4" />

        <div className="inner">
          {/* HERO */}
          <header className="hero">
            <h1 className="hero-title">Movie <em>Recommender</em></h1>
            <p className="hero-sub">Rate 15 films — our model learns purely from your taste, no tracking.</p>
          </header>

          {/* ... (the rest of your component remains exactly the same) ... */}


          {/* PROGRESS */}
          <div className="progress-wrap">
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{
                  width: `${progress}%`,
                  '--dot-opacity': rated.length > 0 ? '1' : '0',
                } as React.CSSProperties}
              />
            </div>
            <span className={`progress-label${rated.length === 15 ? ' done' : ''}`}>
              {rated.length} / 15
            </span>
          </div>

          {/* CARD */}
          <div className="card">

            {rated.length < 15 && (
              <div className="search-wrap">
                <div className="search-field">
                  <span className="search-icon">🔍</span>
                  <input
                    className="search-input"
                    placeholder="Search by title…"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    autoComplete="off"
                  />
                </div>
                {results.length > 0 && (
                  <div className="dropdown">
                    {results.slice(0, 8).map(m => (
                      <button key={m.movie_id} className="dropdown-item" onClick={() => addMovie(m)}>
                        <div className="dropdown-item-icon">🎞</div>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <span className="dropdown-item-title">{m.title}</span>
                          <div className="dropdown-item-meta">
                            {m.year && <span className="tag">{m.year}</span>}
                            {m.genres && <span className="tag">{m.genres}</span>}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {rated.length === 0 ? (
              <div className="empty-state">
                <div className="empty-reel">🎞</div>
                <p className="empty-text">
                  Search for films you've seen<br />and start building your taste profile
                </p>
              </div>
            ) : (
              <div className="movie-list">
                {rated.map((r, i) => (
                  <div key={r.movie_id} className="movie-row">
                    <span className="movie-index">{i + 1}</span>
                    <div className="movie-info">
                      <p className="movie-title">{r.title}</p>
                      <div className="stars">
                        {STARS.map(s => (
                          <button
                            key={s}
                            className={`star-btn ${s <= r.rating ? 'active' : ''}`}
                            onClick={() => setRating(r.movie_id, s)}
                          >★</button>
                        ))}
                      </div>
                    </div>
                    <button className="remove-btn" onClick={() => removeMovie(r.movie_id)}>✕</button>
                  </div>
                ))}
              </div>
            )}

            {error && (
              <div className="error-box">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            )}

            {rated.length > 0 && <div className="card-divider" />}

            <div className="submit-area">
              <button
                className={`submit-btn ${rated.length === 15 && !loading ? 'ready' : 'disabled'}`}
                onClick={submit}
                disabled={rated.length !== 15 || loading}
              >
                {loading
                  ? '✨ Getting your recommendations…'
                  : rated.length === 15
                    ? '🎯 Get My Recommendations →'
                    : `Add ${15 - rated.length} more film${15 - rated.length !== 1 ? 's' : ''} to continue`}
              </button>
            </div>
          </div>

          {/* Footer */}
          <nav className="footer-links">
            <button className="link-btn link-primary" onClick={() => router.push('/similar')}>
              🔍 Find similar movies
            </button>
            <div className="footer-divider" />
            <button className="link-btn link-secondary" onClick={() => router.push('/rate')}>
              Looking for classics?
            </button>
          </nav>

        </div>
      </main>
    </>
  );
}
