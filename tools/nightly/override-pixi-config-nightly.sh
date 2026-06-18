# Remove exclude-newer rule to install scipp pre-release version (100.0.0.dev0).
# Pixi allows overriding package specific exclude-newer rule
# but it does not seem to be working with our scipp nightly build release at the moment.
# This workaround will not be needed once that feature works as expected...
sed -i 's/exclude-newer\ =\ "14d"//' pixi.toml
cat tools/nightly/nightly-pixi.toml >> pixi.toml

