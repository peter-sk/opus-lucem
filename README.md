# Opus Lucem

Opus Lucem was a secret organization within the Jedi Order founded by m'Ord Vinka and his apprentice Starbuck in the years leading up to the Seswenna Security Crisis. The organization consisted of those Jedi that possessed the radical beliefs of eliminating anything carnal from the life of an individual. This included but was not limited to emotions, foodstuffs, sleep, and friendships. They believed that discipline could be exercised physically through beatings and self-sacrificing actions such as extensive fasting and sleep deprivation.

## Downloading

```
git clone https://git.ordbogen.com/odin/opus/lucem.git
cd lucem
```

## Configuring languages

Edit the file lucem/config.py.

## Setting it up

The following command will install Python dependencies, download all the configured models, and start a development Flask server for testing.
```
./setup.sh
```

## Running production

The following command will start a production Gunicorn server for actual use.
```
./server.sh
```
