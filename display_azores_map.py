import folium

# Coordinates for the center of the Azores
azores_coords = [38.5, -28.0]

# Create a map centered on the Azores
m = folium.Map(location=azores_coords, zoom_start=6)

# Add a marker for the Azores
folium.Marker(
    location=azores_coords,
    popup="Azores",
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(m)

# Save the map to an HTML file
m.save("azores_map.html")