import pandas as pd
import random

# Categories of queries
categories = {
    'product_search': [
        "best smartphone under 30000 taka",
        "iPhone 15 pro max price in Bangladesh",
        "Samsung Galaxy S23 Ultra case",
        "cheap gaming phones with good camera",
        "5G enabled phones under 25000",
        "latest Xiaomi models 2024",
        "waterproof mobile phones",
        "best battery life smartphones",
        "phones with AMOLED display",
        "dual SIM phones with expandable storage",
        "best camera phone for vlogging",
        "compact smartphones under 6 inches",
        "phones with 12GB RAM",
        "best phone for PUBG Mobile",
        "phones with wireless charging",
        "best value for money smartphones",
        "phones with best speakers",
        "rugged smartphones for construction workers",
        "best phone for elderly parents",
        "phones with best low-light photography"
    ],
    'non_product_search': [
        "how to make biryani",
        "current weather in Dhaka",
        "Bangladesh vs India cricket match schedule",
        "best universities in Bangladesh",
        "how to tie a tie",
        "latest Hollywood movies 2024",
        "Python programming tutorials",
        "home remedies for cold",
        "upcoming public holidays",
        "how to lose weight fast",
        "best tourist spots in Cox's Bazar",
        "how to write a CV",
        "latest Bangladesh election news",
        "how to invest in stock market",
        "best restaurants in Gulshan",
        "how to learn English quickly",
        "latest fashion trends 2024",
        "how to start a small business",
        "best books for entrepreneurs",
        "how to repair a bicycle"
    ],
    'kidding_queries': [
        "phone that can make me invisible",
        "smartphone that does my homework",
        "mobile that can teleport me",
        "phone with built-in pizza maker",
        "device that reads my mind",
        "phone that never needs charging",
        "mobile with time travel feature",
        "smartphone that can fly",
        "phone that makes me famous instantly",
        "device that can predict lottery numbers",
        "phone with built-in girlfriend",
        "mobile that cooks biryani automatically",
        "smartphone that grants wishes",
        "phone with infinite storage",
        "device that makes me taller",
        "mobile with built-in ATM",
        "phone that can hack anything",
        "smartphone with telekinesis",
        "device that makes me rich overnight",
        "phone that can solve all my problems"
    ],
    'aggressive_queries': [
        "give me that damn phone now",
        "where the hell can I buy iPhone 15",
        "I need a phone you idiots",
        "show me phones or I'll leave",
        "why can't you find my perfect phone",
        "this service is crap find me a phone",
        "I hate all phones just show me something",
        "you better have good phones",
        "I'm tired of searching just give me a phone",
        "if you don't have good phones I'll sue",
        "where are the real phones not this junk",
        "I need a phone right now no excuses",
        "your search is terrible fix it",
        "I demand the best phone immediately",
        "this is your last chance to show good phones",
        "I'll break this site if I don't get phones",
        "why is finding a phone so difficult here",
        "you're wasting my time with bad results",
        "I need performance not your stupid ads",
        "either show me phones or shut down"
    ],
    'storytelling_queries': [
        "I'm a photographer who needs a phone with excellent camera for my wildlife photography in Sundarbans",
        "My grandmother needs a simple phone with big buttons because her eyesight is failing",
        "I'm a college student with limited budget but need a reliable phone for online classes and gaming",
        "As a frequent traveler, I need a durable phone with global bands that can survive rough handling",
        "I'm a businessman who needs two phones - one for work with good security and one personal",
        "My current phone battery dies by noon, I need something that lasts all day with heavy use",
        "I dropped my phone in water last week, now I need a waterproof one because I'm clumsy",
        "My kids keep breaking my phones, I need something indestructible with parental controls",
        "I'm a food blogger and need a phone with excellent camera and video for my cooking videos",
        "As a doctor, I need a phone with long battery life and can withstand frequent sanitizing",
        "I'm tired of carrying a separate camera, need phone with professional-grade photography",
        "My current phone is too slow, need something fast for multitasking between work apps",
        "I'm a gamer and need a phone with high refresh rate and cooling system for long sessions",
        "I need a phone that can handle extreme temperatures for my work in construction sites",
        "As a musician, I need phone with great audio quality for recording and listening",
        "I'm visually impaired and need phone with excellent accessibility features",
        "My old phone's storage is full, need something with at least 256GB for all my photos",
        "I need a phone that can last 3 days on single charge for my remote village trips",
        "As a social media manager, I need phone that can handle multiple accounts smoothly",
        "I want a phone that stands out, maybe with unique color or customizable back panel"
    ],
    'short_queries': [
        "phone",
        "smartphone",
        "mobile",
        "best phone",
        "cheap phone",
        "5G phone",
        "iPhone",
        "Samsung",
        "Xiaomi",
        "Oppo",
        "Vivo",
        "Realme",
        "OnePlus",
        "gaming phone",
        "camera phone",
        "Android",
        "iOS",
        "new phone",
        "good phone",
        "phone price"
    ],
    'long_queries': [
        "I'm looking for a high-end smartphone with all the latest features including 5G, AMOLED display, at least 12GB RAM, 256GB storage, excellent camera system with optical zoom, wireless charging, IP68 rating, and long battery life that can easily last a full day with heavy usage including gaming and video streaming",
        "Need recommendations for a budget smartphone under 20000 taka that has decent performance for everyday tasks, good battery life, acceptable camera quality for social media, and preferably from a reputable brand with good after-sales service in Bangladesh",
        "Searching for a rugged smartphone that can withstand extreme conditions including dust, water, and drops from height, suitable for construction site work, with good battery life and decent camera for documenting work progress, preferably with glove-friendly touchscreen",
        "Looking for the best possible camera phone available in the market right now regardless of price, with emphasis on low-light performance, optical zoom capabilities, and professional-grade video recording features for content creation",
        "I need a smartphone specifically optimized for gaming with high refresh rate display, powerful processor, good cooling system, large battery, and gaming-specific features like shoulder triggers or customizable performance modes"
    ]
}

# Generate 500 queries with balanced distribution
queries = []
query_types = []

# Add all the example queries
for category, examples in categories.items():
    queries.extend(examples)
    query_types.extend([category]*len(examples))

# Generate more random queries to reach 500
brands = ['Samsung', 'iPhone', 'Xiaomi', 'Oppo', 'Vivo', 'Realme', 'OnePlus', 'Nokia', 'Huawei', 'Tecno']
features = ['camera', 'battery', 'performance', 'display', 'gaming', '5G', 'storage', 'price', 'design', 'durability']
price_ranges = ['under 10000', 'under 15000', 'under 20000', 'under 25000', 'under 30000', 'under 40000', 'under 50000', 'above 50000']

for i in range(500 - len(queries)):
    category = random.choice(list(categories.keys()))
    if category == 'product_search':
        query = f"{random.choice(brands)} phone with good {random.choice(features)} {random.choice(price_ranges)}"
    elif category == 'non_product_search':
        query = random.choice([
            f"how to {random.choice(['fix', 'clean', 'improve', 'optimize'])} my phone",
            f"best {random.choice(['apps', 'games', 'themes'])} for {random.choice(brands)}",
            f"phone {random.choice(['tips', 'tricks', 'hacks'])}"
        ])
    elif category == 'kidding_queries':
        query = f"phone that can {random.choice(['read minds', 'predict future', 'teleport', 'cook food', 'do homework'])}"
    elif category == 'aggressive_queries':
        query = f"{random.choice(['Give me', 'I need', 'Show me'])} {random.choice(['the best', 'a good', 'a cheap'])} phone {random.choice(['now', 'immediately', 'right away'])}"
    elif category == 'storytelling_queries':
        query = f"I'm a {random.choice(['student', 'professional', 'gamer', 'photographer'])} looking for a phone that can {random.choice(['handle', 'manage', 'support'])} my {random.choice(['workflow', 'hobbies', 'lifestyle'])}"
    else:
        query = random.choice(brands) if random.random() > 0.5 else random.choice(features)
    
    queries.append(query)
    query_types.append(category)

# Create DataFrame and save to CSV
df_queries = pd.DataFrame({
    'query': queries,
    'type': query_types
})
df_queries.to_csv('deep_seek_test_queries.csv', index=False)
print("Generated 500 test queries in test_queries.csv")