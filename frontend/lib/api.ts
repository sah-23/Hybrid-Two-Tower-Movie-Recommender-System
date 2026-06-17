import axios from 'axios';

const API = axios.create({ baseURL: 'http://localhost:8000/api' });

export interface Movie {
  movie_id: number;
  title: string;
  genres: string;
  year: number | null;
}

export interface RatedMovie {
  movie_id: number;
  rating: number;
  order: number;
}


export interface Demographics {
  gender: 'M' | 'F';
  age: number;
  occupation: number;
}


export interface Recommendation {
  movie_id: number;
  title: string;
  genres: string;
  year: number | null;
  predicted_rating: number;
}


export const searchMovies = async (q: string): Promise<Movie[]> => {
  const res = await API.get('/movies/', { params: { q } });
  return res.data;
};

export const getRecommendations = async (
  demographics: Demographics,
  ratings: RatedMovie[]
): Promise<Recommendation[]> => {
  const res = await API.post('/recommend/', { demographics, ratings });
  return res.data.recommendations;
};


export const searchMovies25m = async (q: string): Promise<Movie[]> => {
  const res = await API.get('/v2/movies/', { params: { q } });
  return res.data;
};

export const getRecommendations25m = async (
  ratings: RatedMovie[]
): Promise<Recommendation[]> => {
  const res = await API.post('/v2/recommend/', { ratings });
  return res.data.recommendations;
};

export interface SimilarMovie {
  movie_id: number;
  title: string;
  genres: string;
  year: number | null;
  similarity: number;
}

export interface SimilarMoviesResult {
  query: { movie_id: number; title: string; genres: string };
  similar: SimilarMovie[];
}

export const getSimilarMovies = async (
  q: string, n = 20
): Promise<SimilarMoviesResult> => {
  const res = await API.get('/similar/', { params: { q, n } });
  return res.data;
};


export const searchMoviesTMDB = (q: string) =>
      API.get('/tmdb/search/', { params: { q } }).then(r => r.data);
export const getSimilarTMDB = (q: string, n = 20) =>
      API.get('/tmdb/similar/', { params: { q, n } }).then(r => r.data);