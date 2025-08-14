#!/usr/bin/env python3
"""
Comparison of different semantic search strategies
"""

class SemanticSearchStrategies:
    """
    Different approaches for semantic search in hybrid systems
    """
    
    def __init__(self, db_helper):
        self.db_helper = db_helper
    
    def strategy_1_full_query(self, query: str, limit: int = 10) -> list:
        """
        Current approach: Use full original query for semantic search
        
        Pros:
        - Preserves complete context and meaning
        - Better for complex, multi-part questions
        - Captures question intent and relationships
        
        Cons:
        - May include less relevant terms
        - Stop words might dilute the search
        """
        query_embedding = self.db_helper.generate_embedding(query)
        return self.db_helper.similarity_search(query_embedding, limit=limit)
    
    def strategy_2_keywords_only(self, keywords: list, limit: int = 10) -> list:
        """
        Alternative: Use only LLM-extracted keywords for semantic search
        
        Pros:
        - More focused on relevant terms
        - Eliminates noise from stop words
        - Consistent with keyword search approach
        
        Cons:
        - Loses contextual relationships
        - May miss important nuances
        - Question structure is lost
        """
        keywords_query = " ".join(keywords)
        query_embedding = self.db_helper.generate_embedding(keywords_query)
        return self.db_helper.similarity_search(query_embedding, limit=limit)
    
    def strategy_3_enhanced_query(self, query: str, keywords: list, entities: list, limit: int = 10) -> list:
        """
        Hybrid approach: Enhance original query with important extracted terms
        
        Pros:
        - Combines benefits of both approaches
        - Emphasizes important terms while preserving context
        - Can boost relevance of key concepts
        
        Cons:
        - More complex to implement
        - May create artificial emphasis
        """
        # Enhance query by emphasizing important terms
        important_terms = keywords + entities
        enhanced_query = f"{query} {' '.join(important_terms)}"
        
        query_embedding = self.db_helper.generate_embedding(enhanced_query)
        return self.db_helper.similarity_search(query_embedding, limit=limit)
    
    def strategy_4_multiple_embeddings(self, query: str, keywords: list, limit: int = 10) -> list:
        """
        Advanced approach: Generate multiple embeddings and combine results
        
        Pros:
        - Captures both specific terms and overall meaning
        - More comprehensive coverage
        - Can weight different aspects differently
        
        Cons:
        - Higher computational cost
        - Complex result merging required
        - May return too many results
        """
        # Get results from full query
        full_query_embedding = self.db_helper.generate_embedding(query)
        full_results = self.db_helper.similarity_search(full_query_embedding, limit=limit//2)
        
        # Get results from keywords
        keywords_query = " ".join(keywords)
        keywords_embedding = self.db_helper.generate_embedding(keywords_query)
        keyword_results = self.db_helper.similarity_search(keywords_embedding, limit=limit//2)
        
        # Merge and deduplicate results
        combined_results = self._merge_semantic_results(full_results, keyword_results, limit)
        return combined_results
    
    def _merge_semantic_results(self, results1: list, results2: list, limit: int) -> list:
        """Helper to merge multiple semantic search results"""
        seen_ids = set()
        merged = []
        
        # Add results from both searches, avoiding duplicates
        for result in results1 + results2:
            chunk_id = result.get('id')
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(result)
                if len(merged) >= limit:
                    break
        
        return merged

# Example usage and comparison
def compare_strategies():
    """
    Compare different semantic search strategies with example queries
    """
    
    test_cases = [
        {
            "query": "How does the chunking algorithm work with document processing?",
            "keywords": ["chunking", "algorithm", "document", "processing"],
            "entities": [],
            "expected_focus": "Process and methodology"
        },
        {
            "query": "What database tables store the vector embeddings?",
            "keywords": ["database", "tables", "vector", "embeddings"],
            "entities": ["PostgreSQL"],
            "expected_focus": "Technical implementation details"
        },
        {
            "query": "Explain the difference between semantic and keyword search",
            "keywords": ["semantic", "keyword", "search", "difference"],
            "entities": [],
            "expected_focus": "Conceptual comparison"
        }
    ]
    
    print("🔍 Semantic Search Strategy Comparison")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['query']}")
        print(f"🏷️  Keywords: {test_case['keywords']}")
        print(f"🎯 Expected Focus: {test_case['expected_focus']}")
        print("-" * 40)
        
        # Strategy comparisons would go here
        print("Strategy 1 (Full Query): Preserves complete question context")
        print("Strategy 2 (Keywords Only): Focuses on specific terms")
        print("Strategy 3 (Enhanced): Balances context with emphasis")
        print("Strategy 4 (Multiple): Comprehensive but complex")

if __name__ == "__main__":
    compare_strategies()
