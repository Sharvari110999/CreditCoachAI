"""Main entry point for the Credit System."""

from src.creditsystem.engine import create_engine


def main():
    """Main entry point for the Credit System."""
    engine = create_engine()
    
    print("Credit System Engine initialized.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        question = input("You: ")
        
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not question.strip():
            continue
        
        try:
            response = engine.process_question(question)
            print(f"\n{response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
