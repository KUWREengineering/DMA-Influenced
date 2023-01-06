var map = L.map('map');
var tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxNativeZoom: 18,
    maxZoom: 20
});
var geoPipes = L.geoJson({{pipe_geom.to_json()|safe}}, pipeOptions);
var geoNodes = L.geoJson({{node_geom.to_json()|safe}}, nodeOptions);

function renderPipe(layer) {
  if (layer.feature.properties.highlight) {
    layer.setStyle({color: "cyan"});
  }
  else {
    layer.setStyle(pipeOptions);
  }
}

geoPipes
  .eachLayer(function(layer) {
    renderPipe(layer);
  })
  .on('mouseover', function(ev) {
    ev.layer.setStyle({color:"yellow"});
  })
  .on('mouseout', function(ev) {
    renderPipe(ev.layer);
  })
  .bindTooltip(function(layer) {
    return '<b>Pipe: </b>' + layer.feature.id;
  })
  .on('click', function(e) {
    var url_stub = "{{ url_for('pipe_details', id='XXX') }}";
    window.location = url_stub.replace("XXX", e.layer.feature.id);
  });

geoNodes
  .bindTooltip(function(layer) {
    return '<b>Node: </b>' + layer.feature.id;
  })
  .on('click', function(e) {
    var url_stub = "{{ url_for('node_details', id='XXX') }}";
    window.location = url_stub.replace("XXX", e.layer.feature.id);
  });

tileLayer.setOpacity(0.5);
tileLayer.addTo(map);
geoPipes.addTo(map);
geoNodes.addTo(map);
map.fitBounds(L.featureGroup([geoPipes, geoNodes]).getBounds());
