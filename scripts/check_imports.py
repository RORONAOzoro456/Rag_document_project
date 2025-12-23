names = [
    'langchain_core.documents',
    'langchain_core',
    'langchain_community.document_loaders',
    'langchain_community',
    'langchain_text_splitters',
    'langchain_text_splitters.splitters',
    'langchain_core.prompts',
]
for n in names:
    try:
        __import__(n)
        print(n, 'FOUND')
    except Exception as e:
        print(n, 'MISSING', type(e).__name__)
