"""
Template definitions for numerical encoding evaluation.
Contains text templates and context definitions.
"""

from typing import Dict, List

def create_text_templates() -> Dict[str, List[str]]:
    """Create templates for generating contextual number representations.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping context types to template lists
    """
    return {
        'rating': [
            'rated {} stars',
            'gave it {} points',
            'score of {}'
        ],
        'price': [
            'costs ${}.00',
            'priced at ${}.99',
            '${} total'
        ],
        'quantity': [
            '{} items',
            '{} units available',
            'quantity: {}'
        ],
        'time': [
            '{} hours',
            '{} minutes',
            '{} days old'
        ],
        'percentage': [
            '{}% complete',
            '{}% increase',
            'grew by {}%'
        ]
    }

def get_context_templates(context_type: str) -> List[str]:
    """Get templates for a specific context type.
    
    Args:
        context_type: Type of context ('rating', 'price', etc.)
        
    Returns:
        List[str]: List of templates for the context
    """
    templates = create_text_templates()
    return templates.get(context_type, ['{} {}'])

def get_all_context_types() -> List[str]:
    """Get list of all available context types.
    
    Returns:
        List[str]: List of context type names
    """
    return list(create_text_templates().keys())