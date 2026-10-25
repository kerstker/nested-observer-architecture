from datetime import datetime
import random
from src.core import clock

class EnvironmentSensor:
    def __init__(self):
        self.location = "Virtual Room"
        
    def get_context(self) -> dict:
        now = datetime.fromtimestamp(clock.now())
        hour = now.hour
        
        # 1. Day/Night Cycle
        is_night = hour < 6 or hour > 20
        time_desc = "Night" if is_night else "Day"
        
        # 2. Simulated Weather (Simple random for now, could integrate API)
        # Seed by hour to keep it consistent within the hour.
        # Use a private generator: seeding the global `random` here made every other
        # random draw in the system (idle thoughts, actions) repeat within the hour.
        rng = random.Random(now.strftime("%Y%m%d%H"))
        weathers = ["Sunny", "Rainy", "Cloudy", "Stormy"]
        weather = rng.choice(weathers)
        
        # 3. Festivals (Simple check)
        date_str = now.strftime("%m-%d")
        festival = None
        if date_str == "01-01": festival = "New Year"
        elif date_str == "12-25": festival = "Christmas"
        elif date_str == "02-14": festival = "Valentine's Day"
        
        return {
            "time": now.strftime("%H:%M"),
            "period": time_desc,
            "weather": weather,
            "festival": festival,
            "is_night": is_night
        }
