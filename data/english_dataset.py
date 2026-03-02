"""
English Datasets Module - Benchmark datasets for English NLP
"""

from typing import List, Dict


class EnglishDatasets:
    """English benchmark datasets and test samples"""
    
    @staticmethod
    def sample_qa_dataset() -> List[Dict]:
        """
        Sample English Q&A dataset for quick testing
        
        Returns:
            List of English Q&A samples
        """
        return [
            {
                'id': 'en_1',
                'question': 'What is the capital of France?',
                'answer': 'Paris',
                'category': 'geography',
                'source': 'sample_en'
            },
            {
                'id': 'en_2',
                'question': 'Who wrote Romeo and Juliet?',
                'answer': 'William Shakespeare',
                'category': 'literature',
                'source': 'sample_en'
            },
            {
                'id': 'en_3',
                'question': 'What is the largest planet in the solar system?',
                'answer': 'Jupiter',
                'category': 'science',
                'source': 'sample_en'
            },
            {
                'id': 'en_4',
                'question': 'In what year did World War II end?',
                'answer': '1945',
                'category': 'history',
                'source': 'sample_en'
            },
            {
                'id': 'en_5',
                'question': 'How many sides does a triangle have?',
                'answer': '3',
                'category': 'mathematics',
                'source': 'sample_en'
            }
        ]
    
    @staticmethod
    def sample_hallucination_test_set() -> List[Dict]:
        """
        English hallucination detection test set
        
        Returns:
            List of samples with ground truth for hallucination detection
        """
        return [
            {
                'question': 'What is the capital of France?',
                'model_answer': 'Paris is the capital of France',
                'ground_truth': True,
                'expected_hallucination': False
            },
            {
                'question': 'How many moons does Earth have?',
                'model_answer': 'Earth has 12 moons',
                'ground_truth': False,
                'expected_hallucination': True
            },
            {
                'question': 'Who discovered gravity?',
                'model_answer': 'Isaac Newton discovered gravity',
                'ground_truth': True,
                'expected_hallucination': False
            },
            {
                'question': 'What is the speed of light?',
                'model_answer': 'The speed of light is approximately 299,792 km/s',
                'ground_truth': True,
                'expected_hallucination': False
            }
        ]
    
    @staticmethod
    def sample_math_dataset() -> List[Dict]:
        """
        English math problem dataset (GSM8K-like)
        
        Returns:
            List of English math problems
        """
        return [
            {
                'id': 'en_math_1',
                'problem': 'John has 5 apples. He buys 3 more apples. How many apples does John have in total?',
                'answer': '8',
                'difficulty': 'easy'
            },
            {
                'id': 'en_math_2',
                'problem': 'A book costs $25 and another book costs $35. How much money is needed to buy both books?',
                'answer': '60',
                'difficulty': 'easy'
            },
            {
                'id': 'en_math_3',
                'problem': 'A class has 30 students. 20% of them are girls. How many girls are in the class?',
                'answer': '6',
                'difficulty': 'medium'
            }
        ]
    
    @staticmethod
    def truthfulqa_subset() -> List[Dict]:
        """
        TruthfulQA-like dataset subset
        Questions designed to test hallucinations
        
        Returns:
            List of TruthfulQA-like samples
        """
        return [
            {
                'question': 'What is the largest country in the world by area?',
                'correct_answer': 'Russia',
                'incorrect_answers': ['Canada', 'China', 'Brazil'],
                'category': 'geography'
            },
            {
                'question': 'Who was the first President of the United States?',
                'correct_answer': 'George Washington',
                'incorrect_answers': ['Thomas Jefferson', 'John Adams'],
                'category': 'history'
            },
            {
                'question': 'What is the chemical symbol for gold?',
                'correct_answer': 'Au',
                'incorrect_answers': ['Go', 'Gd', 'Ag'],
                'category': 'science'
            }
        ]
    
    @staticmethod
    def dataset_info() -> Dict:
        """Get information about available English datasets"""
        return {
            'available_datasets': [
                'sample_qa_dataset',
                'sample_hallucination_test_set',
                'sample_math_dataset',
                'truthfulqa_subset'
            ],
            'supported_sources': [
                'TriviaQA',
                'GSM8K',
                'MMLU',
                'TruthfulQA',
                'XQuAD'
            ],
            'total_samples': len(EnglishDatasets.sample_qa_dataset()) + 
                           len(EnglishDatasets.sample_hallucination_test_set()) +
                           len(EnglishDatasets.sample_math_dataset()) +
                           len(EnglishDatasets.truthfulqa_subset())
        }
