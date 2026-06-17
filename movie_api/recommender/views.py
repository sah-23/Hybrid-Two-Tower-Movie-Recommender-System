from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from . import inference
from .ml.predictor import get_predictor


@api_view(['GET'])
def movies_list(request):
    q = request.GET.get('q', '').strip()
    if q:
        # starts-with results come first, then contains — cap at 20
        all_movies = get_predictor().get_all_movies()
        q_lower    = q.lower()
        starts     = [m for m in all_movies if m['title'].lower().startswith(q_lower)]
        contains   = [m for m in all_movies if not m['title'].lower().startswith(q_lower)
                      and q_lower in m['title'].lower()]
        return Response((starts + contains)[:20])
    # no query → return popular movies for initial display
    n = int(request.GET.get('n', 50))
    return Response(inference.get_popular_movies(n))


@api_view(['POST'])
def recommend(request):
    data = request.data
    try:
        demographics = data['demographics']
        # accept either field name from the frontend
        rated_movies = data.get('rated_movies') or data.get('ratings', [])
        top_n        = int(data.get('top_n', 20))

        if len(rated_movies) != 15:
            return Response(
                {'error': f'Exactly 15 rated movies required, got {len(rated_movies)}'},
                status=status.HTTP_400_BAD_REQUEST
            )

        results = inference.recommend(demographics, rated_movies, top_n=top_n)
        return Response({'recommendations': results})

    except KeyError as e:
        return Response({'error': f'Missing field: {e}'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def movies_list_25m(request):
    from . import inference_25m
    q = request.GET.get('q', '').strip()
    if q:
        return Response(inference_25m.search_movies(q, limit=20))
    return Response(inference_25m.get_popular_movies(50))


import traceback
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

@api_view(['POST'])
def recommend_25m(request):
    from . import inference_25m
    data = request.data
    try:
        rated_movies = data.get('rated_movies') or data.get('ratings', [])
        top_n        = int(data.get('top_n', 20))
        
        if len(rated_movies) != 15:
            return Response(
                {'error': f'Exactly 15 rated movies required, got {len(rated_movies)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        results = inference_25m.recommend(rated_movies, top_n=top_n)
        return Response({'recommendations': results})
        
    except Exception as e:
        # Print the full error stack trace to your terminal/server logs
        print("--- CRASH DURING INFERENCE ---")
        traceback.print_exc()
        
        # Also return it in the API response so you can see it in Postman/browser
        return Response(
            {
                'error': str(e),
                'traceback': traceback.format_exc()
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )



@api_view(['GET'])
def similar_movies(request):
    """
    /api/similar/?q=The+Matrix&n=20
    Returns movies with similar content to the query title.
    Uses cosine similarity over SBERT embeddings stored in the predictor.
    """
    q     = request.GET.get('q', '').strip()
    top_n = int(request.GET.get('n', 20))

    if not q:
        return Response({'error': 'q parameter required'},
                        status=status.HTTP_400_BAD_REQUEST)
    try:
        result = get_predictor().similar_movies(q, top_n=top_n)
        if 'error' in result:
            return Response(result, status=status.HTTP_404_NOT_FOUND)
        return Response(result)
    except Exception as e:
        return Response({'error': str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

    
@api_view(['GET'])
def tmdb_search(request):
    from . import inference_tmdb
    q = request.GET.get('q', '').strip()
    if not q: return Response([])
    return Response(inference_tmdb.search_movies(q))



@api_view(['GET'])
def tmdb_similar(request):
    from . import inference_tmdb
    q = request.GET.get('q', '').strip()
    #request.GET is the query parameters from the URL.
    if not q: return Response({'error': 'q required'}, status=400)
    result = inference_tmdb.find_similar(q, top_n=int(request.GET.get('n', 20)))
    if 'error' in result: return Response(result, status=404)
    return Response(result)