import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { PrimeNGConfig } from 'primeng/api';
import {SwUpdate} from '@angular/service-worker';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'Banco de sangre';
  public constructor(
    private primengConfig: PrimeNGConfig, private cdRef: ChangeDetectorRef, private updates: SwUpdate) {
   }

  ngOnInit() {
    this.primengConfig.ripple = true;
    this.actualizaciones();
  }

  updateChecked = false;
  updateAvailable = false;

  async actualizaciones() {
    this.updates.available.subscribe(() => {
      // Keep the loading indicator active while we reload the page
      this.updateAvailable = true;
      window.location.reload();
    });
    if (this.updates.isEnabled) {
      // This promise will return when the update check is completed,
      // and if an update is available, it will also wait for the update
      // to be downloaded (at which point it calls our callback above and
      // we just need to reload the page to apply it).
      await this.updates.checkForUpdate();
    } else {
      console.log('Service worker updates are disabled.');
    }
    // The update check is done (or service workers are disabled), now
    // we can take the loading indicator down (unless we need to apply an
    // update, but in that case, we have already set this.updateAvailable
    // to true by this point, which keeps the loading indicator active).
    this.updateChecked = true;
  }
}
